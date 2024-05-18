  ''' Functions in HDR flow '''

import os
import cv2 as cv
import numpy as np

Z_max = 255
Z_min = 0
gamma = 2.2

def ReadImg(path, flag=1):
    img = cv.imread(path, flag)  # flag = 1 means to load a color image
    img = img[:,:,[2,1,0]]
    return img


def SaveImg(img, path):
    img = img[:,:,[2,1,0]]
    cv.imwrite(path, img)
    

def LoadExposures(source_dir):
    """ load bracketing images folder

    Args:
        source_dir (string): folder path containing bracketing images and a image_list.txt file
                             image_list.txt contains lines of image_file_name, exposure time, ... 
    Returns:
        img_list (uint8 ndarray, shape (N, height, width, ch)): N bracketing images (3 channel)
        exposure_times (list of float, size N): N exposure times
    """
    filenames = []
    exposure_times = []
    f = open(os.path.join(source_dir, 'image_list.txt'))
    for line in f:
        if (line[0] == '#'):
            continue
        (filename, exposure, *_) = line.split()
        filenames += [filename]
        exposure_times += [float(exposure)]
    img_list = [ReadImg(os.path.join(source_dir, f)) for f in filenames]
    img_list = np.array(img_list)
    
    return img_list, exposure_times


def PixelSample(img_list):
    """ Sampling

    Args:
        img_list (uint8 ndarray, shape (N, height, width, ch))
        
    Returns:
        sample (uint8 ndarray, shape (N, height_sample_size, width_sample_size, ch))
    """
    # trivial periodic sample
    sample = img_list[:, ::64, ::64, :]
    
    return sample
    

def EstimateResponse(img_samples, etime_list, lambda_=50):
    """ Estimate camera response for bracketing images

    Args:
        img_samples (uint8 ndarray, shape (N, height_sample_size, width_sample_size)): N bracketing sampled images (1 channel)
        etime_list (list of float, size N): N exposure times
        lambda_ (float): Lagrange multiplier (Defaults to 50)
    
    Returns:
        response (float ndarray, shape (256)): response map
    """
    
    ''' TODO '''
    etime_list = np.array(etime_list, dtype=np.float32)
    img_samples = np.reshape(img_samples, (np.size(img_samples, 0), np.size(img_samples, 1)*np.size(img_samples, 2)))
    
    n = 256
    A = np.zeros((np.size(img_samples, 0) * np.size(img_samples, 1) + n + 1, n + np.size(img_samples, 1)), dtype=np.float32)
    b = np.zeros((np.size(A, 0), 1), dtype=np.float32)
    w = [z if z <= 0.5*Z_max else Z_max-z for z in range(256)]

    k = 0
    for i in range(np.size(img_samples, 1)):
        for j in range(np.size(img_samples, 0)):
            z = int(img_samples[j][i])
            wij = w[z]
            A[k][z] = wij
            A[k][n+i] = -wij
            b[k] = wij * np.log(etime_list[j])
            k += 1
    
    A[k][128] = 1
    k += 1

    for i in range(n-1):
        A[k][i]   =      lambda_ * w[i+1]
        A[k][i+1] = -2 * lambda_ * w[i+1]
        A[k][i+2] =      lambda_ * w[i+1]
        k += 1

    x = np.linalg.lstsq(A, b, rcond=None)[0]
    response = x[:256].reshape(256,)
    #lE = x[256:]
    return response


def ConstructRadiance(img_list, response, etime_list):
    """ Construct radiance map from brackting images

    Args:
        img_list (uint8 ndarray, shape (N, height, width)): N bracketing images (1 channel)
        response (float ndarray, shape (256)): response map
        etime_list (list of float, size N): N exposure times
    
    Returns:
        radiance (float ndarray, shape (height, width)): radiance map
    """

    ''' TODO '''
    etime_list = np.array(etime_list, dtype=np.float32)
    img_list = img_list.astype(np.float32)
    img_size = img_list[0].shape
    w = [z if z <= 0.5*Z_max else Z_max-z for z in range(256)]
    ln_t = np.log(etime_list)

    radiance = np.zeros_like(img_list[0], dtype=np.float32)

    Z = [img.flatten().tolist() for img in img_list]
    acc_E = [0]*len(Z[0])
    ln_E = [0]*len(Z[0])

    vfunc = np.vectorize(lambda x:np.exp(x))
    pixels, imgs = len(Z[0]), len(Z)
    for i in range(pixels):
        acc_w = 0
        for j in range(imgs):
            z = int(Z[j][i])
            acc_E[i] += w[z]*(response[z] - ln_t[j])
            acc_w += w[z]
        ln_E[i] = acc_E[i]/acc_w if acc_w > 0 else acc_E[i]
        acc_w = 0

    # Exponational each channels and reshape to 2D-matrix
    #radiance = np.reshape(np.power(2,ln_E), img_size)
    radiance = np.reshape(vfunc(ln_E), img_size)
    #print('done')

    return radiance

def CameraResponseCalibration(src_path, lambda_):
    img_list, exposure_times = LoadExposures(src_path)
    radiance = np.zeros_like(img_list[0], dtype=np.float32)
    pixel_samples = PixelSample(img_list)
    for ch in range(3):
        response = EstimateResponse(pixel_samples[:, :,:, ch], exposure_times, lambda_)
        radiance[:,:,ch] = ConstructRadiance(img_list[:,:,:,ch], response, exposure_times)

    return radiance


def WhiteBalance(src, y_range, x_range):
    """ White balance based on Known to be White(KTBW) region

    Args:
        src (float ndarray, shape (height, width, ch)): source radiance
        y_range (tuple of 2 int): location range in y-dimension
        x_range (tuple of 2 int): location range in x-dimension
        
    Returns:
        result (float ndarray, shape (height, width, ch))
    """
   
    ''' TODO '''
    result = src.copy()
    ktbw_region = src[y_range[0]:y_range[1], x_range[0]:x_range[1], :]
    avg_white = np.mean(ktbw_region, axis=(0, 1))

    # Compute the scaling factors for each channel
    scale_factors = avg_white / avg_white[0]

    # Apply the scaling factors to the entire image
    result[:, :, 1] /= scale_factors[1]
    result[:, :, 2] /= scale_factors[2]

    return result


def GlobalTM(src, scale=1.0):
    """ Global tone mapping

    Args:
        src (float ndarray, shape (height, width, ch)): source radiance image
        scale (float): scaling factor (Defaults to 1.0)
    
    Returns:
        result(uint8 ndarray, shape (height, width, ch)): result HDR image
    """
    
    ''' TODO '''
    Xmax = np.max(src)
    s = scale
    log_Xmax = np.log2(Xmax)
    
    # Apply tone mapping in log domain
    log_X = np.log2(src)
    log_X_hat = s * (log_X - log_Xmax) + log_Xmax
    
    # Transform back to linear domain
    X_hat = np.power(2, log_X_hat)
    
    # Apply gamma correction
    gamma = 2.2
    X_corrected = np.power(X_hat, 1/gamma)
    
    # Scale to range [0, 255]
    X_scaled = np.clip(X_corrected, 0, 1) * 255
    
    # Round to integers
    result = X_scaled.astype(np.uint8)
    
    return result


def LocalTM(src, imgFilter, scale=3.0):
    """ Local tone mapping

    Args:
        src (float ndarray, shape (height, width,ch)): source radiance image
        imgFilter (function): filter function with preset parameters
        scale (float): scaling factor (Defaults to 3.0)
    
    Returns:
        result(uint8 ndarray, shape (height, width,ch)): result HDR image
    """
    
    ''' TODO '''
    # Step 1: Separate intensity map (I) and color ratio (Cx)
    I = np.mean(src, axis=2) #equation (10)
    Cx = src / I[:,:,np.newaxis] #equation (11)

    # Step 2: Take log of intensity
    L = np.log2(I)

    # Step 3: Separate detail layer (LD) and base layer (LB) of L with Gaussian filter
    LB = imgFilter(L)
    LD = L - LB

    # Step 4: Compress the contrast of base layer
    Lmin = np.min(LB)
    Lmax = np.max(LB)
    LB_compressed = ((LB - Lmax) * scale) / (Lmax - Lmin)

    # Step 5: Reconstruct intensity map with adjusted base layer and detail layer
    I_compressed = 2**(LB_compressed + LD)

    # Step 6: Reconstruct color map with adjusted intensity and color ratio
    result = Cx * I_compressed[:,:,np.newaxis]

    # Gamma correction
    gamma = 2.2
    result = np.power(result, 1/gamma)

    # Scale to [0, 255]
    result = np.clip(result, 0, 1) * 255
    result = result.astype(np.uint8)

    return result

def histogram_equalization(src):
    """Histogram Equalization

    Args:
        src (numpy.ndarray): Input grayscale image

    Returns:
        dst (numpy.ndarray): Output equalized image
    """
    # Compute histogram
    hist, _ = np.histogram(src.flatten(), bins=256, range=(0, 256))

    # Compute CDF
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()

    # Equalization mapping
    equalized_values = np.round(cdf_normalized * 255).astype(np.uint8)

    # Apply mapping to the image
    dst = np.interp(src.flatten(), np.arange(256), equalized_values).reshape(src.shape)

    return dst

def GaussianFilter(src, N=35, sigma_s=100):
    """ Gaussian filter

    Args:
        src (float ndarray, shape (height, width)): source intensity
        N (int): window size of the filter (Defaults to 35)
                 filter indices span [-N/2, N/2]
        sigma_s (float): standard deviation of Gaussian filter (Defaults to 100)
    
    Returns:
        result (float ndarray, shape (height, width))
    """
    
    ''' TODO '''
     # Create Gaussian kernel
    kernel = np.zeros((N, N))
    center = N // 2
    for i in range(N):
        for j in range(N):
            kernel[i, j] = np.exp(-((i - center)**2 + (j - center)**2) / (2 * sigma_s**2))
    kernel /= np.sum(kernel)

    # Apply the filter to the source image
    result = np.zeros_like(src)
    height, width = src.shape
    pad_size = N // 2
    padded_src = np.pad(src, pad_size, mode='reflect')
    for i in range(height):
        for j in range(width):
            result[i, j] = np.sum(padded_src[i:i+N, j:j+N] * kernel)

    return result


def BilateralFilter(src, N=35, sigma_s=100, sigma_r=0.8):
    """ Bilateral filter

    Args:
        src (float ndarray, shape (height, width)): source intensity
        N (int): window size of the filter (Defaults to 35)
                 filter indices span [-N/2, N/2]
        sigma_s (float): spatial standard deviation of bilateral filter (Defaults to 100)
        sigma_r (float): range standard deviation of bilateral filter (Defaults to 0.8)
    
    Returns:
        result (float ndarray, shape (height, width))
    """
    
    ''' TODO '''

    # Get image shape
    height, width = src.shape

    # Initialize result
    result = np.zeros((height, width), dtype=np.float32)

    # Pre-compute spatial component
    spatial_filter = np.zeros((N, N), dtype=np.float32)
    for i in range(-N//2, N//2+1):
        for j in range(-N//2, N//2+1):
            spatial_filter[i+N//2, j+N//2] = np.exp(-(i**2 + j**2) / (2 * sigma_s**2))

    # Iterate over each pixel in the image
    for i in range(height):
        for j in range(width):
            intensity = src[i, j]

            # Define window boundaries
            i_min = max(i - N//2, 0)
            i_max = min(i + N//2, height-1)
            j_min = max(j - N//2, 0)
            j_max = min(j + N//2, width-1)

            # Compute the range component
            range_filter = np.exp(-((src[i_min:i_max+1, j_min:j_max+1] - intensity)**2) / (2 * sigma_r**2))

            # Compute the bilateral filter response
            bilateral_filter = spatial_filter[i_min-i+N//2:i_max-i+N//2+1, j_min-j+N//2:j_max-j+N//2+1] * range_filter
            result[i, j] = np.sum(bilateral_filter * src[i_min:i_max+1, j_min:j_max+1]) / np.sum(bilateral_filter)

    return result
