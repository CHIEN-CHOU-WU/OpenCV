1. Crop image to a multiple of blk_size
2. Rrandomly replace each tile in the original image with one of five color-channel versions of it.
3. Randomly create gradient tiles and replace some of the blocks in the image with them
4. Create the puzzle version of the image
5. Modify the image in a fashion similar to part #2, but replace each image block/tile by a filtered version of the block, instead of a different color-channel version of the block.
6. Combine everything, to create a version.

To run the script:
python image_manipulation and_filtering.py images/cat.jpg 100