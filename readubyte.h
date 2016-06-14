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

#ifndef __CUDNN_TRAINING_READUBYTE_H
#define __CUDNN_TRAINING_READUBYTE_H

#include <cstdint>
#include <cstddef>

/**
 * Obtains images and labels from a UByte dataset. If "data" and "labels" are null,
 * only the number of images is returned.
 * 
 * @param image_filename The dataset file containing the images.
 * @param label_filename The dataset file containing the labels.
 * @param data The output dataset, a Dx1xHxW array (single channel, D is the dataset size).
 * @param labels The Dx1 label array.
 * @param width The width of each image.
 * @param height The height of each image.
 * @return Number of images in dataset.
 */
size_t ReadUByteDataset(const char* image_filename, const char* label_filename, 
                        uint8_t *data, uint8_t *labels, size_t& width, size_t& height);

#endif  // __CUDNN_TRAINING_READUBYTE_H
