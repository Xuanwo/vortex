// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Locating the compressed page bodies inside a Parquet file.
//!
//! Parquet compresses each page body independently with a block codec, which is exactly the
//! shape nvCOMP's batched decompression entrypoints consume: an array of independent
//! compressed chunks with known uncompressed sizes. This module finds those chunks so the
//! GPU backend can hand the whole batch to the device in one launch, the same decomposition
//! cuDF's Parquet reader uses.
//!
//! Column chunk ranges come from the file footer; page boundaries within a chunk are only
//! discoverable by walking the per-page Thrift headers, so a minimal Thrift compact-protocol
//! reader lives here. `parquet::format::PageHeader` is deprecated and scheduled for removal,
//! and `parquet`'s own page-header parser is crate-private, so neither can be used.

use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use clap::ValueEnum;
use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::file::metadata::ParquetMetaData;
use parquet::file::properties::EnabledStatistics;
use parquet::file::properties::WriterProperties;
use parquet::file::properties::WriterVersion;

/// Target size of a data page written for GPU decompression.
///
/// nvCOMP decompresses one chunk per page, so pages must be large enough to amortize the
/// per-chunk setup yet numerous enough to fill the device. ~1 MiB is the page size cuDF's
/// Parquet reader is tuned around.
pub const GPU_DATA_PAGE_SIZE: usize = 1024 * 1024;

/// Row cap per data page.
///
/// `parquet`'s default caps pages at 20k rows, which produces pages far below
/// [`GPU_DATA_PAGE_SIZE`] for narrow columns and leaves the device underfed.
const GPU_DATA_PAGE_ROW_COUNT_LIMIT: usize = 1_000_000;

/// Parquet page codecs that nvCOMP can decompress on the device.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum GpuCodec {
    /// The Parquet default, and the codec with the highest device-side throughput.
    #[default]
    Snappy,
    /// Matches the codec used by the CPU Parquet benchmark, at lower device throughput.
    Zstd,
}

impl GpuCodec {
    /// The Parquet compression setting for this codec.
    pub fn to_parquet(self) -> Compression {
        match self {
            GpuCodec::Snappy => Compression::SNAPPY,
            GpuCodec::Zstd => Compression::ZSTD(ZstdLevel::default()),
        }
    }

    /// Short lowercase name, used in measurement labels.
    pub fn name(self) -> &'static str {
        match self {
            GpuCodec::Snappy => "snappy",
            GpuCodec::Zstd => "zstd",
        }
    }

    /// Whether a column chunk's codec matches this one.
    pub fn matches(self, compression: Compression) -> bool {
        matches!(
            (self, compression),
            (GpuCodec::Snappy, Compression::SNAPPY) | (GpuCodec::Zstd, Compression::ZSTD(_))
        )
    }

    /// Decompress a single page body on the host, for cross-checking device output.
    pub fn decompress_host(self, compressed: &[u8], uncompressed_len: usize) -> Result<Vec<u8>> {
        let decompressed = match self {
            GpuCodec::Snappy => snap::raw::Decoder::new().decompress_vec(compressed)?,
            GpuCodec::Zstd => zstd::bulk::decompress(compressed, uncompressed_len)?,
        };
        ensure!(
            decompressed.len() == uncompressed_len,
            "page decompressed to {} bytes, page header declared {uncompressed_len}",
            decompressed.len()
        );
        Ok(decompressed)
    }
}

/// Writer properties tuned for GPU decompression.
pub fn gpu_writer_properties(codec: GpuCodec) -> WriterProperties {
    WriterProperties::builder()
        // V1 data pages compress the entire page body, which is the unit nvCOMP decompresses.
        // V2 pages place uncompressed repetition/definition levels ahead of the compressed
        // values inside one page body, which the batched entrypoints cannot express.
        .set_writer_version(WriterVersion::PARQUET_1_0)
        .set_compression(codec.to_parquet())
        // Dictionary encoding keeps the decompressed payload small and is the encoding GPU
        // Parquet readers decode fastest.
        .set_dictionary_enabled(true)
        .set_data_page_size_limit(GPU_DATA_PAGE_SIZE)
        .set_data_page_row_count_limit(GPU_DATA_PAGE_ROW_COUNT_LIMIT)
        // Per-page statistics only inflate the page headers that have to be walked on the host.
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .build()
}

/// A compressed page body located within a Parquet file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompressedPage {
    /// Offset of the compressed body, i.e. just past the page header.
    pub offset: usize,
    /// Length of the compressed body in bytes.
    pub compressed_len: usize,
    /// Length of the body once decompressed.
    pub uncompressed_len: usize,
}

/// The pages of one column chunk, alongside the byte range the chunk occupies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnChunkPages {
    /// Offset of the column chunk within the file.
    pub offset: u64,
    /// Length of the column chunk in bytes.
    pub len: usize,
    /// Compressed pages of this chunk, in file order, with file-absolute offsets.
    pub pages: Vec<CompressedPage>,
}

/// Walks every column chunk's page headers and returns the compressed page bodies.
///
/// The outer `Vec` is one entry per row group, which is the unit a reader can stage on the
/// device and release before moving on. Chunks and the pages within them are in file order.
pub fn scan_compressed_pages(
    file_bytes: &[u8],
    metadata: &ParquetMetaData,
) -> Result<Vec<Vec<ColumnChunkPages>>> {
    let mut row_groups = Vec::with_capacity(metadata.row_groups().len());

    for row_group in metadata.row_groups() {
        let mut chunks = Vec::with_capacity(row_group.columns().len());
        for column in row_group.columns() {
            let (chunk_offset, chunk_len) = column.byte_range();
            let mut pages = Vec::new();
            let (start, len) = (chunk_offset, chunk_len);
            let start = usize::try_from(start)?;
            let end = start
                .checked_add(usize::try_from(len)?)
                .filter(|end| *end <= file_bytes.len())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "column chunk range {start}..+{len} extends past the {} byte file",
                        file_bytes.len()
                    )
                })?;

            let mut pos = start;
            while pos < end {
                let header = read_page_header(&file_bytes[pos..end])?;
                let body = pos + header.header_len;
                let body_end = body
                    .checked_add(header.compressed_len)
                    .filter(|body_end| *body_end <= end)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "page body at {body} of {} bytes overruns its column chunk",
                            header.compressed_len
                        )
                    })?;

                match header.page_type {
                    PageType::Data | PageType::Dictionary => pages.push(CompressedPage {
                        offset: body,
                        compressed_len: header.compressed_len,
                        uncompressed_len: header.uncompressed_len,
                    }),
                    PageType::DataV2 => bail!(
                        "v2 data pages are not GPU-decompressible as a single chunk; \
                         write the file with WriterVersion::PARQUET_1_0"
                    ),
                    // Index pages are not part of the column data and are never written by
                    // `parquet`; skip over the body rather than decompressing it.
                    PageType::Index => {}
                }

                pos = body_end;
            }

            chunks.push(ColumnChunkPages {
                offset: chunk_offset,
                len: usize::try_from(chunk_len)?,
                pages,
            });
        }

        row_groups.push(chunks);
    }

    Ok(row_groups)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PageType {
    Data,
    Index,
    Dictionary,
    DataV2,
}

struct PageHeaderInfo {
    header_len: usize,
    page_type: PageType,
    compressed_len: usize,
    uncompressed_len: usize,
}

/// Thrift compact-protocol field types.
mod ttype {
    pub(super) const STOP: u8 = 0x00;
    pub(super) const BOOL_TRUE: u8 = 0x01;
    pub(super) const BOOL_FALSE: u8 = 0x02;
    pub(super) const I8: u8 = 0x03;
    pub(super) const I16: u8 = 0x04;
    pub(super) const I32: u8 = 0x05;
    pub(super) const I64: u8 = 0x06;
    pub(super) const DOUBLE: u8 = 0x07;
    pub(super) const BINARY: u8 = 0x08;
    pub(super) const LIST: u8 = 0x09;
    pub(super) const SET: u8 = 0x0a;
    pub(super) const MAP: u8 = 0x0b;
    pub(super) const STRUCT: u8 = 0x0c;
    pub(super) const UUID: u8 = 0x0d;
}

/// Guards against unbounded recursion on malformed headers.
const MAX_STRUCT_DEPTH: u32 = 32;

/// Reads the `PageHeader` at the start of `buf`, returning its fields and encoded length.
fn read_page_header(buf: &[u8]) -> Result<PageHeaderInfo> {
    let mut reader = CompactReader { buf, pos: 0 };
    let mut page_type = None;
    let mut uncompressed_len = None;
    let mut compressed_len = None;
    let mut last_field_id = 0i16;

    while let Some((field_id, field_type)) = reader.read_field_header(&mut last_field_id)? {
        match (field_id, field_type) {
            (1, ttype::I32) => page_type = Some(reader.read_i32()?),
            (2, ttype::I32) => uncompressed_len = Some(reader.read_i32()?),
            (3, ttype::I32) => compressed_len = Some(reader.read_i32()?),
            _ => reader.skip_value(field_type, 0)?,
        }
    }

    let page_type = match page_type {
        Some(0) => PageType::Data,
        Some(1) => PageType::Index,
        Some(2) => PageType::Dictionary,
        Some(3) => PageType::DataV2,
        Some(other) => bail!("unknown Parquet page type {other}"),
        None => bail!("page header is missing its page type"),
    };

    let uncompressed_len =
        uncompressed_len.ok_or_else(|| anyhow::anyhow!("page header is missing its size"))?;
    let compressed_len = compressed_len
        .ok_or_else(|| anyhow::anyhow!("page header is missing its compressed size"))?;

    Ok(PageHeaderInfo {
        header_len: reader.pos,
        page_type,
        compressed_len: usize::try_from(compressed_len)?,
        uncompressed_len: usize::try_from(uncompressed_len)?,
    })
}

/// Minimal reader for the subset of the Thrift compact protocol that page headers use.
struct CompactReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl CompactReader<'_> {
    fn read_u8(&mut self) -> Result<u8> {
        let byte = *self
            .buf
            .get(self.pos)
            .ok_or_else(|| anyhow::anyhow!("page header ends mid-field"))?;
        self.pos += 1;
        Ok(byte)
    }

    fn advance(&mut self, len: usize) -> Result<()> {
        let end = self
            .pos
            .checked_add(len)
            .filter(|end| *end <= self.buf.len())
            .ok_or_else(|| anyhow::anyhow!("page header ends mid-value"))?;
        self.pos = end;
        Ok(())
    }

    fn read_varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = self.read_u8()?;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        bail!("varint in page header is not terminated")
    }

    fn read_zigzag(&mut self) -> Result<i64> {
        let encoded = self.read_varint()?;
        Ok(((encoded >> 1) as i64) ^ -((encoded & 1) as i64))
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(i32::try_from(self.read_zigzag()?)?)
    }

    /// Reads the next field header, or `None` at the struct's STOP byte.
    fn read_field_header(&mut self, last_field_id: &mut i16) -> Result<Option<(i16, u8)>> {
        let header = self.read_u8()?;
        if header == ttype::STOP {
            return Ok(None);
        }

        let field_type = header & 0x0f;
        let delta = header >> 4;
        let field_id = if delta == 0 {
            i16::try_from(self.read_zigzag()?)?
        } else {
            last_field_id
                .checked_add(i16::from(delta))
                .ok_or_else(|| anyhow::anyhow!("field id overflow in page header"))?
        };
        *last_field_id = field_id;

        Ok(Some((field_id, field_type)))
    }

    fn skip_struct(&mut self, depth: u32) -> Result<()> {
        ensure!(
            depth < MAX_STRUCT_DEPTH,
            "page header nests structs more than {MAX_STRUCT_DEPTH} deep"
        );
        let mut last_field_id = 0i16;
        while let Some((_, field_type)) = self.read_field_header(&mut last_field_id)? {
            self.skip_value(field_type, depth + 1)?;
        }
        Ok(())
    }

    /// Skips a field value. Booleans carry their value in the field type, so consume nothing.
    fn skip_value(&mut self, field_type: u8, depth: u32) -> Result<()> {
        match field_type {
            ttype::BOOL_TRUE | ttype::BOOL_FALSE => Ok(()),
            ttype::I8 => self.advance(1),
            ttype::I16 | ttype::I32 | ttype::I64 => self.read_varint().map(|_| ()),
            ttype::DOUBLE => self.advance(8),
            ttype::UUID => self.advance(16),
            ttype::BINARY => {
                let len = usize::try_from(self.read_varint()?)?;
                self.advance(len)
            }
            ttype::LIST | ttype::SET => {
                let (len, element_type) = self.read_collection_header()?;
                for _ in 0..len {
                    self.skip_element(element_type, depth + 1)?;
                }
                Ok(())
            }
            ttype::MAP => {
                let len = usize::try_from(self.read_varint()?)?;
                if len > 0 {
                    let types = self.read_u8()?;
                    let (key_type, value_type) = (types >> 4, types & 0x0f);
                    for _ in 0..len {
                        self.skip_element(key_type, depth + 1)?;
                        self.skip_element(value_type, depth + 1)?;
                    }
                }
                Ok(())
            }
            ttype::STRUCT => self.skip_struct(depth),
            other => bail!("unsupported Thrift compact type {other} in page header"),
        }
    }

    /// Skips one collection element. Unlike fields, booleans here occupy a byte of their own.
    fn skip_element(&mut self, element_type: u8, depth: u32) -> Result<()> {
        match element_type {
            ttype::BOOL_TRUE | ttype::BOOL_FALSE => self.advance(1),
            other => self.skip_value(other, depth),
        }
    }

    fn read_collection_header(&mut self) -> Result<(usize, u8)> {
        let header = self.read_u8()?;
        let element_type = header & 0x0f;
        let len = match header >> 4 {
            0x0f => usize::try_from(self.read_varint()?)?,
            short_len => usize::from(short_len),
        };
        Ok((len, element_type))
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::sync::Arc;

    use arrow_array::Int64Array;
    use arrow_array::RecordBatch;
    use arrow_array::StringArray;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use parquet::arrow::ArrowWriter;
    use parquet::file::metadata::ParquetMetaDataReader;
    use parquet::file::reader::FileReader;
    use parquet::file::reader::SerializedFileReader;
    use rstest::rstest;

    use super::*;

    fn sample_batch() -> Result<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("ints", DataType::Int64, false),
            Field::new("strings", DataType::Utf8, false),
        ]));
        let ints = Int64Array::from_iter_values((0..50_000).map(|i| i % 977));
        let strings =
            StringArray::from_iter_values((0..50_000).map(|i| format!("value-{}", i % 1_000)));
        Ok(RecordBatch::try_new(
            schema,
            vec![Arc::new(ints), Arc::new(strings)],
        )?)
    }

    fn write_sample(path: &std::path::Path, codec: GpuCodec) -> Result<()> {
        let batch = sample_batch()?;
        let file = File::create(path)?;
        let mut writer =
            ArrowWriter::try_new(file, batch.schema(), Some(gpu_writer_properties(codec)))?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }

    /// The page bodies we locate must decompress to exactly the bytes `parquet` itself reads.
    #[rstest]
    #[case(GpuCodec::Snappy)]
    #[case(GpuCodec::Zstd)]
    fn scanned_pages_match_parquet_reader(#[case] codec: GpuCodec) -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("sample.parquet");
        write_sample(&path, codec)?;

        let file = File::open(&path)?;
        let metadata = ParquetMetaDataReader::new().parse_and_finish(&file)?;
        let file_bytes = std::fs::read(&path)?;
        let row_groups = scan_compressed_pages(&file_bytes, &metadata)?;
        let pages = row_groups
            .iter()
            .flatten()
            .flat_map(|chunk| chunk.pages.iter())
            .collect::<Vec<_>>();

        let reader = SerializedFileReader::new(File::open(&path)?)?;
        let mut expected = Vec::new();
        for row_group in 0..reader.metadata().num_row_groups() {
            let row_group_reader = reader.get_row_group(row_group)?;
            for column in 0..row_group_reader.num_columns() {
                let mut page_reader = row_group_reader.get_column_page_reader(column)?;
                while let Some(page) = page_reader.get_next_page()? {
                    expected.push(page.buffer().to_vec());
                }
            }
        }

        assert_eq!(pages.len(), expected.len(), "page count mismatch");
        assert!(!pages.is_empty(), "expected the sample file to have pages");

        for (page, expected) in pages.iter().zip(expected.iter()) {
            let compressed = &file_bytes[page.offset..page.offset + page.compressed_len];
            let decompressed = codec.decompress_host(compressed, page.uncompressed_len)?;
            assert_eq!(&decompressed, expected);
        }

        Ok(())
    }

    /// Pages must tile their column chunks exactly, with no gap left unaccounted for.
    #[test]
    fn scanned_pages_cover_every_column_chunk() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("sample.parquet");
        write_sample(&path, GpuCodec::Snappy)?;

        let file = File::open(&path)?;
        let metadata = ParquetMetaDataReader::new().parse_and_finish(&file)?;
        let file_bytes = std::fs::read(&path)?;
        let row_groups = scan_compressed_pages(&file_bytes, &metadata)?;
        let pages = row_groups
            .iter()
            .flatten()
            .flat_map(|chunk| chunk.pages.iter())
            .collect::<Vec<_>>();

        let compressed: usize = pages.iter().map(|page| page.compressed_len).sum();
        let chunk_total: i64 = metadata
            .row_groups()
            .iter()
            .flat_map(|rg| rg.columns())
            .map(|col| col.compressed_size())
            .sum();

        // The chunk total includes the page headers, so the page bodies must be strictly
        // smaller but within a header's worth per page.
        assert!(compressed < usize::try_from(chunk_total)?);
        assert!(compressed > usize::try_from(chunk_total)? - pages.len() * 256);
        Ok(())
    }
}
