/*
 * This code is released into the public domain.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include "readubyte.h"

#include <cstdio>
#include <cstdlib>

#define UBYTE_IMAGE_MAGIC 2051
#define UBYTE_LABEL_MAGIC 2049

#ifdef _MSC_VER
    #define bswap(x) _byteswap_ulong(x)
#else
    #define bswap(x) __builtin_bswap32(x)
#endif

#pragma pack(push, 1)
struct UByteImageDataset 
{
    /// Magic number (UBYTE_IMAGE_MAGIC).
    uint32_t magic;

    /// Number of images in dataset.
    uint32_t length;

    /// The height of each image.
    uint32_t height;

    /// The width of each image.
    uint32_t width;

    void Swap()
    {
        magic = bswap(magic);
        length = bswap(length);
        height = bswap(height);
        width = bswap(width);
    }
};

struct UByteLabelDataset
{
    /// Magic number (UBYTE_LABEL_MAGIC).
    uint32_t magic;

    /// Number of labels in dataset.
    uint32_t length;

    void Swap()
    {
        magic = bswap(magic);
        length = bswap(length);
    }
};
#pragma pack(pop)

size_t ReadUByteDataset(const char *image_filename, const char *label_filename,
                        uint8_t *data, uint8_t *labels, size_t& width, size_t& height)
{
    FILE *imfp = fopen(image_filename, "rb");
    if (!imfp)
    {
        printf("ERROR: Cannot open image dataset %s\n", image_filename);
        return 0;
    }
    FILE *lbfp = fopen(label_filename, "rb");
    if (!lbfp)
    {
        fclose(imfp);
        printf("ERROR: Cannot open label dataset %s\n", label_filename);
        return 0;
    }

    UByteImageDataset image_header;
    UByteLabelDataset label_header;
    
    // Read and verify file headers
    if (fread(&image_header, sizeof(UByteImageDataset), 1, imfp) != 1)
    {
        printf("ERROR: Invalid dataset file (image file header)\n");
        fclose(imfp);
        fclose(lbfp);
        return 0;
    }
    if (fread(&label_header, sizeof(UByteLabelDataset), 1, lbfp) != 1)
    {
        printf("ERROR: Invalid dataset file (label file header)\n");
        fclose(imfp);
        fclose(lbfp);
        return 0;
    }

    // Byte-swap data structure values (change endianness)
    image_header.Swap();
    label_header.Swap();

    // Verify datasets
    if (image_header.magic != UBYTE_IMAGE_MAGIC)
    {
        printf("ERROR: Invalid dataset file (image file magic number)\n");
        fclose(imfp);
        fclose(lbfp);
        return 0;
    }
    if (label_header.magic != UBYTE_LABEL_MAGIC)
    {
        printf("ERROR: Invalid dataset file (label file magic number)\n");
        fclose(imfp);
        fclose(lbfp);
        return 0;
    }
    if (image_header.length != label_header.length)
    {
        printf("ERROR: Dataset file mismatch (number of images does not match the number of labels)\n");
        fclose(imfp);
        fclose(lbfp);
        return 0;
    }
    
    // Output dimensions
    width = image_header.width;
    height = image_header.height;

    // Read images and labels (if requested)
    if (data != nullptr)
    {
        if (fread(data, sizeof(uint8_t), image_header.length * width * height, imfp) != image_header.length * width * height)
        {
            printf("ERROR: Invalid dataset file (partial image dataset)\n");
            fclose(imfp);
            fclose(lbfp);
            return 0;
        }
    }
    if (labels != nullptr)
    {
        if (fread(labels, sizeof(uint8_t), label_header.length, lbfp) != label_header.length)
        {
            printf("ERROR: Invalid dataset file (partial label dataset)\n");
            fclose(imfp);
            fclose(lbfp);
            return 0;
        }
    }
    
    fclose(imfp);
    fclose(lbfp);

    return image_header.length;
}
