
import cv2
import numpy as np
import random as rand
import sys

def main():
    
    # 1) Modify the function for getting two arguments from the command line
    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

    possibility = 0.33
    possibility_filter = 0.33

    # check args
    if len(args) != 2:
        print('Please enter name (and path) and block size.')
        return 0
    
    try:

        img_name = args[0]          # Image name (and path)
        blk_size = int(args[1])     # Block size
    except:
        print('Wrong agrs, try again.')
        return 0

    rand.seed()

    img = cv2.imread (img_name)

    # crop image to a multiple of blk_size
    h_extra = img.shape[0] % blk_size
    w_extra = img.shape[1] % blk_size

    crop_img = img[0:(img.shape[0]-h_extra),0:(img.shape[1]-w_extra)]

    cv2.imshow ('Crop image', crop_img)

    # compute num horz & vert blks
    h_blks = crop_img.shape[0] / blk_size
    w_blks = crop_img.shape[1] / blk_size

    tiled_img = crop_img.copy()

    # 2) Loop through all the blocks and randomly select one version of that block.
    for r in range(int(h_blks)):
        for c in range(int(w_blks)):
            blk_img = img[r * blk_size : r *blk_size+blk_size,
                c * blk_size : c * blk_size + blk_size]

            rand_block = getRandColorBlock(blk_img)

            tiled_img[blk_size*r:blk_size*r+blk_size,
                  blk_size*c:blk_size*c+blk_size] = rand_block  #blk_img
    
    cv2.imshow ('Tiled image random version', tiled_img)

    # 3) Randomly create gradient tiles and replace some of the blocks in the image with them
    tiled_img_gradient = crop_img.copy()

    for r in range(int(h_blks)):
        for c in range(int(w_blks)):
            # 1/3 chance to replace it with gradient color block
            if rand.random() < possibility:
                img_color_grad = getGradientBlock(blk_size)
                tiled_img_gradient[blk_size*r:blk_size*r+blk_size, blk_size*c:blk_size*c+blk_size] = img_color_grad  #blk_img

    cv2.imshow ('Tiled image random gradient (Prob = 1/3)', tiled_img_gradient)

    # 4) Create hte puzzle version of the image
    img_puzzle = crop_img.copy()
    img_puzzle = createPuzzleImage(img_puzzle)

    cv2.imshow ('Puzzle image', img_puzzle)

    # 5) Replace each image block/tile by a filtered version of the block
    tiled_filter_img = crop_img.copy()
    for r in range(int(h_blks)):
        for c in range(int(w_blks)):
            blk_img = img[r * blk_size : r *blk_size+blk_size,
                c * blk_size : c * blk_size + blk_size]
            rand_filter_block = getFilteredBlock(blk_img)
            tiled_filter_img[blk_size*r:blk_size*r+blk_size, blk_size*c:blk_size*c+blk_size] = rand_filter_block

    cv2.imshow ('Filtered image', tiled_filter_img)


    # 6) Combine everything
    combination = crop_img.copy()
    # Create a tile puzzle
    combination = createPuzzleImage(combination)
    tiled_img_combination = crop_img.copy()
    for r in range(int(h_blks)):
        for c in range(int(w_blks)):
            blk_img = combination[r * blk_size : r *blk_size+blk_size,
                c * blk_size : c * blk_size + blk_size]

            rand_block = getRandColorBlock(blk_img)
            # Replaces some of the blocks with gradient color blocks (part #3)
            if rand.random() < possibility:
                img_color_grad = getGradientBlock(blk_size)
                tiled_img_combination[blk_size*r:blk_size*r+blk_size, blk_size*c:blk_size*c+blk_size] = img_color_grad
            
            # Modifies the color-channel version for each tile (part #2)
            else:
                
                tiled_img_combination[blk_size*r:blk_size*r+blk_size,
                  blk_size*c:blk_size*c+blk_size] = rand_block  #blk_img
            
            # Applies random filters to each tile (part #5)
            if rand.random() < possibility_filter:
                rand_filter_block = getFilteredBlock(rand_block)
                tiled_img_combination[blk_size*r:blk_size*r+blk_size, blk_size*c:blk_size*c+blk_size] = rand_filter_block


    cv2.imshow ('Combination image', tiled_img_combination)
    

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getRandColorBlock(img_block):
    # show blue, green, red, gray and BGR reversed as RGB components
    
    # Blue
    b_blk = img_block.copy()
    b_blk[:,:,1] = 0
    b_blk[:,:,2] = 0

    # Green
    g_blk = img_block.copy()
    g_blk[:,:,0] = 0
    g_blk[:,:,2] = 0

    # Red
    r_blk = img_block.copy()
    r_blk[:,:,0] = 0
    r_blk[:,:,1] = 0

    # Gray
    gray_blk_2d = img_block.copy()
    gray_blk_2d = cv2.cvtColor(gray_blk_2d, cv2.COLOR_BGR2GRAY)
    gray_blk = np.zeros_like(img_block)
    gray_blk[:, :, 0] = gray_blk_2d
    gray_blk[:, :, 1] = gray_blk_2d
    gray_blk[:, :, 2] = gray_blk_2d

    # BGR reversed as RGB
    rgb_img = img_block.copy()
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    # using a list, enable random selection from among the four tile options
    blk_list = [img_block, b_blk, g_blk, r_blk, gray_blk, rgb_img]
    rand_blk = rand.randrange (len(blk_list))

    return blk_list[rand_blk]

def getGradientBlock(block_size):

    # Randomly select one of the possible 8 directions
    directions = ['l-r', 'r-l', 't-b', 'b-t', 'tl-br', 'br-tl', 'tr-bl', 'bl-tr']
    rand_dir = rand.randrange(8)
    direction = directions[rand_dir]

    # Randomly select a color
    colors = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan']
    rand_color = rand.randrange(6)
    color = colors[rand_color]

    # Randomly choose whether the opposite side of the gradient is going to be white or black
    bws = ['black', 'white']
    rand_bw = rand.randrange(2)
    bw = bws[rand_bw]

    # create images as numpy arrays
    img_color = np.zeros( (block_size,block_size,3), dtype=np.uint8)

    if color == 'red':
        img_color[:,:,2] = 255
        img_color_grad = img_color.copy()
        channel_1 = 2
        channel_2 = 0
        channel_3 = 1

    if color == 'green':
        img_color[:,:,1] = 255
        img_color_grad = img_color.copy()
        channel_1 = 1
        channel_2 = 0
        channel_3 = 2
    
    if color == 'blue':
        img_color[:,:,0] = 255
        img_color_grad = img_color.copy()
        channel_1 = 0
        channel_2 = 1
        channel_3 = 2

    if color == 'red' or color == 'green' or color == 'blue':
        if direction == 'l-r': 
            for c in range(block_size):
                if bw == 'black':
                    img_color_grad[:, c, channel_1] = 255 - 255/block_size * c
                if bw == 'white':
                    img_color_grad[:, c, channel_2] = 255/block_size * c
                    img_color_grad[:, c, channel_3] = 255/block_size * c
        
        if direction == 'r-l':
            for c in range(block_size):
                if bw == 'black':
                    img_color_grad[:, c, channel_1] = 255 - 255/block_size * (block_size - c)
                if bw == 'white':
                    img_color_grad[:, c, channel_2] = 255/block_size * (block_size - c)
                    img_color_grad[:, c, channel_3] = 255/block_size * (block_size - c)

        if direction == 't-b':
            for r in range(block_size):
                if bw == 'black':
                    img_color_grad[r, :, channel_1] = 255 - 255/block_size * r
                if bw == 'white':
                    img_color_grad[r, :, channel_2] = 255/block_size * r
                    img_color_grad[r, :, channel_3] = 255/block_size * r 
        
        if direction == 'b-t':
            for r in range(block_size):
                if bw == 'black':
                    img_color_grad[r, :, channel_1] = 255 - 255/block_size * (block_size - r)
                if bw == 'white':
                    img_color_grad[r, :, channel_2] = 255/block_size * (block_size - r)
                    img_color_grad[r, :, channel_3] = 255/block_size * (block_size - r)

        if direction == 'tl-br':
            for r in range(block_size):
                for c in range(block_size):
                    if bw == 'black':
                        img_color_grad[r,c,channel_1] = 255 - 255/(block_size * 2) * ((r+c))
                    if bw == 'white':
                        img_color_grad[r,c,channel_2] = 255/(block_size * 2) * (r+c)
                        img_color_grad[r,c,channel_3] = 255/(block_size * 2) * (r+c)

        if direction == 'br-tl':
            for r in range(block_size):
                for c in range(block_size):
                    if bw == 'black':
                        img_color_grad[r,c,channel_1] = 255 - 255/(block_size * 2) * (2 * block_size - (r+c))
                    if bw == 'white':
                        img_color_grad[r,c,channel_2] = 255/(block_size * 2) * (2 * block_size - (r+c))
                        img_color_grad[r,c,channel_3] = 255/(block_size * 2) * (2 * block_size - (r+c))
        
        if direction == 'tr-bl':
            for r in range(block_size):
                for c in range(block_size):
                    if bw == 'black':
                        img_color_grad[r,c,channel_1] = 255 - 255/(block_size * 2) * ((r+ block_size - c))
                    if bw == 'white':                
                        img_color_grad[r,c,channel_2] = 255/(block_size * 2) * (r+ block_size - c)
                        img_color_grad[r,c,channel_3] = 255/(block_size * 2) * (r+ block_size - c)

        if direction == 'bl-tr':
            for r in range(block_size):
                for c in range(block_size):
                    if bw == 'black':
                        img_color_grad[r,c,channel_1] = 255 - 255/(block_size * 2) * (2 * block_size - (r+ block_size - c))
                    if bw == 'white':
                        img_color_grad[r,c,channel_2] = 255/(block_size * 2) * (2 * block_size - (r+ block_size - c))
                        img_color_grad[r,c,channel_3] = 255/(block_size * 2) * (2 * block_size - (r+ block_size - c))

    if color == 'yellow':

        img_color_grad = img_color.copy()
        if bw == 'white':
            img_color_grad[:,:] = 255   # fully white image
        channel_1 = 1
        channel_2 = 2
        channel_3 = 0
    
    if color == 'magenta':

        img_color_grad = img_color.copy()
        if bw == 'white':
            img_color_grad[:,:] = 255   # fully white image
        channel_1 = 0
        channel_2 = 2
        channel_3 = 1

    if color == 'cyan':

        img_color_grad = img_color.copy()
        if bw == 'white':
            img_color_grad[:,:] = 255   # fully white image
        channel_1 = 0
        channel_2 = 1
        channel_3 = 2
        
    if color == 'yellow' or color == 'magenta' or color == 'cyan':
        if direction == 'l-r':
            for c in range(block_size):
                if bw == 'black':
                    img_color_grad[:,c,channel_1] = 255 - 255/block_size * c
                    img_color_grad[:,c,channel_2] = 255 - 255/block_size * c
                if bw == 'white':
                    img_color_grad[:,c,channel_3] = 255/block_size * c

        if direction == 'r-l':
            for c in range(block_size):
                if bw == 'black':
                    img_color_grad[:,c,channel_1] = 255/block_size * c
                    img_color_grad[:,c,channel_2] = 255/block_size * c
                if bw == 'white':
                    img_color_grad[:,c,channel_3] = 255 - 255/block_size * c

        if direction == 't-b':
            for r in range(block_size):
                if bw == 'black':
                    img_color_grad[r,:,channel_1] = 255 - 255/block_size * r
                    img_color_grad[r,:,channel_2] = 255 - 255/block_size * r
                if bw == 'white':
                    img_color_grad[r,:,channel_3] = 255/block_size * r

        if direction == 'b-t':
            for r in range(block_size):
                if bw == 'black':
                    img_color_grad[r,:,channel_1] = 255/block_size * r
                    img_color_grad[r,:,channel_2] = 255/block_size * r
                if bw == 'white':
                    img_color_grad[r,:,channel_3] = 255 - 255/block_size * r

        if direction == 'tl-br':
            for r in range(block_size):
                for c in range(block_size):
                    if bw == 'black':
                        img_color_grad[r,c,channel_1] = 255 - 255/(block_size * 2) * (r+c)
                        img_color_grad[r,c,channel_2] = 255 - 255/(block_size * 2) * (r+c)
                    if bw == 'white':
                        img_color_grad[r,c,channel_3] = 255/(block_size * 2) * (r+c)

        if direction == 'br-tl':
            for r in range(block_size):
                for c in range(block_size):
                    if bw == 'black':
                        img_color_grad[r,c,channel_1] = 255/(block_size * 2) * (r+c)
                        img_color_grad[r,c,channel_2] = 255/(block_size * 2) * (r+c)
                    if bw == 'white':
                        img_color_grad[r,c,channel_3] = 255 - 255/(block_size * 2) * (r+c)

        if direction == 'tr-bl':
            for r in range(block_size):
                for c in range(block_size):
                    if bw == 'black':
                        img_color_grad[r,c,channel_1] = 255 - 255/(block_size * 2) * ((r+ block_size - c))
                        img_color_grad[r,c,channel_2] = 255 - 255/(block_size * 2) * ((r+ block_size - c))
                    if bw == 'white':
                        img_color_grad[r,c,channel_3] = 255/(block_size * 2) * ((r+ block_size - c))                        

        if direction == 'bl-tr':
            for r in range(block_size):
                for c in range(block_size):
                    if bw == 'black':
                        img_color_grad[r,c,channel_1] = 255/(block_size * 2) * ((r+ block_size - c))
                        img_color_grad[r,c,channel_2] = 255/(block_size * 2) * ((r+ block_size - c))
                    if bw == 'white':
                        img_color_grad[r,c,channel_3] = 255 - 255/(block_size * 2) * ((r+ block_size - c))
    
    return img_color_grad

def createPuzzleImage(img):
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    blk_size = int(args[1])     # Block size

    # crop image to a multiple of blk_size
    h_extra = img.shape[0] % blk_size
    w_extra = img.shape[1] % blk_size

    crop_img = img[0:(img.shape[0]-h_extra),0:(img.shape[1]-w_extra)]
    
    # compute num horz & vert blks
    h_blks = crop_img.shape[0] / blk_size
    w_blks = crop_img.shape[1] / blk_size

    tiled_img = crop_img.copy()
    
    blocks = []
    for r in range(int(h_blks)):
        for c in range(int(w_blks)):
            blk_img = img[r * blk_size : r *blk_size+blk_size,
                c * blk_size : c * blk_size + blk_size]
            
            blocks.append(blk_img)

    rand.shuffle(blocks)

    i = 0
    for r in range(int(h_blks)):
        for c in range(int(w_blks)):
            tiled_img[blk_size*r:blk_size*r+blk_size,
                blk_size*c:blk_size*c+blk_size] = blocks[i]
            i += 1

    return tiled_img

def getFilteredBlock(img_block):
    # Randomly select filter size
    filter_sizes = [3, 5, 7, 9, 11]
    rand_filter_size = rand.randrange(5)
    filter_size = filter_sizes[rand_filter_size]

    # Randomly select sigmax
    sigmaxs = [cv2.BORDER_CONSTANT, cv2.BORDER_DEFAULT, cv2.BORDER_ISOLATED, cv2.BORDER_REFLECT, cv2.BORDER_REFLECT101, cv2.BORDER_REPLICATE, cv2.BORDER_TRANSPARENT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101]
    rand_sigma = rand.randrange(9)
    sigmax = sigmaxs[rand_sigma]

    # Randomly choose one filter
    rand_filter_select = rand.randrange(6)

    if rand_filter_select == 0:
        # box filter
        img_block = cv2.boxFilter(img_block, -1, (filter_size,filter_size))

    if rand_filter_select == 1:
        # # Gaussian blur filter
        img_block = cv2.GaussianBlur (img_block, (filter_size,filter_size), sigmax)

    if rand_filter_select == 2:
        # # median filter
        img_block = cv2.medianBlur(img_block, filter_size)

    if rand_filter_select == 3:
        # Gaussian sharpening filter
        kernel = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]]) - np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])/9
        img_block = cv2.filter2D(img_block, -1, kernel)

    if rand_filter_select == 4:
        # Laplacian filter
        img_block = cv2.Laplacian(img_block, -1, ksize=filter_size)

    if rand_filter_select == 5:
        #Sobel edge filters (one in x-direction, one in y-direction)
        img_block = cv2.Sobel(img_block, -1, 1, 1)

    return img_block

if __name__ == "__main__":
    main()