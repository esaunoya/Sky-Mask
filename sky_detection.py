import cv2
import base64
import numpy as np
from scipy import spatial


def make_mask(b, image):
    mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
    for xx, yy in enumerate(b):
        mask[yy:, xx] = 255

    return mask


def mask_only(b, image, color=[255, 255, 255]):
    result = image.copy()
    overlay = np.full(image.shape, color, image.dtype)

    mask = cv2.addWeighted(
            cv2.bitwise_not(overlay, overlay, mask=make_mask(b, image)),
            1,
            0,
            1,
            0,
            result
        )

    # cv2.imwrite(
    #     'output/mask.jpg',
    #     mask
    # )

    return mask






def no_sky_found_mask(image):

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, mask) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

    # cv2.imwrite('output/mask.jpg', mask)

    return mask




def display_cv2_image(image):
    return cv2.imencode('.png', image)[1].tostring()


# ip = get_ipython()
# png_f = ip.display_formatter.formatters['image/png']
# png_f.for_type_by_name('numpy', 'ndarray', display_cv2_image);

def color_to_gradient(image):
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    return np.hypot(
        cv2.Sobel(gray, cv2.CV_64F, 1, 0),
        cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    )


def energy(b_tmp, image):
    sky_mask = make_mask(b_tmp, image)

    ground = np.ma.array(
        image,
        mask=cv2.cvtColor(cv2.bitwise_not(sky_mask), cv2.COLOR_GRAY2BGR)
    ).compressed()
    sky = np.ma.array(
        image,
        mask=cv2.cvtColor(sky_mask, cv2.COLOR_GRAY2BGR)
    ).compressed()
    ground.shape = (ground.size//3, 3)
    sky.shape = (sky.size//3, 3)

    sigma_g, mu_g = cv2.calcCovarMatrix(
        ground,
        None,
        cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE
    )
    sigma_s, mu_s = cv2.calcCovarMatrix(
        sky,
        None,
        cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE
    )

    y = 2

    return 1 / (
        (y * np.linalg.det(sigma_s) + np.linalg.det(sigma_g)) +
        (y * np.linalg.det(np.linalg.eig(sigma_s)[1]) +
            np.linalg.det(np.linalg.eig(sigma_g)[1]))
    )


def calculate_border(grad, t):
    sky = np.full(grad.shape[1], grad.shape[0])

    for x in range(grad.shape[1]):
        border_pos = np.argmax(grad[:, x] > t)

        # argmax hax return 0 if nothing is > t
        if border_pos > 0:
            sky[x] = border_pos

    return sky


def calculate_border_optimal(image, thresh_min=5, thresh_max=600, search_step=5):
    grad = color_to_gradient(image)

    n = ((thresh_max - thresh_min) // search_step) + 1

    b_opt = None
    jn_max = 0

    for k in range(1, n + 1):
        t = thresh_min + ((thresh_max - thresh_min) // n - 1) * (k - 1)

        b_tmp = calculate_border(grad, t)
        jn = energy(b_tmp, image)

        if jn > jn_max:
            jn_max = jn
            b_opt = b_tmp

    return b_opt


def no_sky_region(bopt, thresh1, thresh2, thresh3):
    border_ave = np.average(bopt)
    asadsbp = np.average(np.absolute(np.diff(bopt)))

    return border_ave < thresh1 or (border_ave < thresh2 and asadsbp > thresh3)


def partial_sky_region(bopt, thresh4):
    return np.any(np.diff(bopt) > thresh4)


def refine_sky(bopt, image):
    sky_mask = make_mask(bopt, image)

    ground = np.ma.array(
        image,
        mask=cv2.cvtColor(cv2.bitwise_not(sky_mask), cv2.COLOR_GRAY2BGR)
    ).compressed()
    sky = np.ma.array(
        image,
        mask=cv2.cvtColor(sky_mask, cv2.COLOR_GRAY2BGR)
    ).compressed()
    ground.shape = (ground.size//3, 3)
    sky.shape = (sky.size//3, 3)

    ret, label, center = cv2.kmeans(
        np.float32(sky),
        2,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    sigma_s1, mu_s1 = cv2.calcCovarMatrix(
        sky[label.ravel() == 0],
        None,
        cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE
    )
    ic_s1 = cv2.invert(sigma_s1, cv2.DECOMP_SVD)[1]

    sigma_s2, mu_s2 = cv2.calcCovarMatrix(
        sky[label.ravel() == 1],
        None,
        cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE
    )
    ic_s2 = cv2.invert(sigma_s2, cv2.DECOMP_SVD)[1]

    sigma_g, mu_g = cv2.calcCovarMatrix(
        ground,
        None,
        cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE
    )
    icg = cv2.invert(sigma_g, cv2.DECOMP_SVD)[1]

    if cv2.Mahalanobis(mu_s1, mu_g, ic_s1) > cv2.Mahalanobis(mu_s2, mu_g, ic_s2):
        mu_s = mu_s1
        sigma_s = sigma_s1
        ics = ic_s1
    else:
        mu_s = mu_s2
        sigma_s = sigma_s2
        ics = ic_s2

    for x in range(image.shape[1]):
        cnt = np.sum(np.less(
            spatial.distance.cdist(
                image[0:bopt[x], x],
                mu_s,
                'mahalanobis',
                VI=ics
            ),
            spatial.distance.cdist(
                image[0:bopt[x], x],
                mu_g,
                'mahalanobis',
                VI=icg
            )
        ))

        if cnt < (bopt[x] / 2):
            bopt[x] = 0

    return bopt


def detect_sky(image):
    # display(input_image)
    # cv2.imwrite('output/input_image.jpg', input_image)

    bopt = calculate_border_optimal(image)

    if no_sky_region(bopt, image.shape[0]/30, image.shape[0]/4, 5):
        # display("No sky detected")
        # print('no sky detected')
        mask = no_sky_found_mask(image)
        return mask

    # display_mask(bopt, image)
    mask = mask_only(bopt, image)

    if partial_sky_region(bopt, image.shape[1]/3):
        bnew = refine_sky(bopt, image)

        # display_mask(bnew, image)
        mask = mask_only(bnew, image)

    return mask


def sky_ratio(image):
    sky = np.sum(image == 255)
    notSky = np.sum(image == 0)
    per = (sky / (sky + notSky))

    # print(f'Image is {per}% sky.')

    return per

def encode(path):
    with open(path, "rb") as f:
        im_b64 = base64.b64encode(f.read())
    return im_b64

def decode(im_b64):
    im_bytes = base64.b64decode(im_b64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return img


# # input_image = cv2.imread("fixtures/full_sky.png")
# input_image = cv2.imread("fixtures/partial_sky.png")
# input_image = cv2.imread("fixtures/no_sky.png")

# input_image = cv2.imread("fixtures/backyard.jpg")


## encoded image we will receive
img_encoded = encode("fixtures/no_sky.png")

## decode the received image
input_image = decode(img_encoded)

## magic
mask = detect_sky(input_image)

_, im_arr = cv2.imencode('.jpg', mask)  # im_arr: image in Numpy one-dim array format.
im_bytes = im_arr.tobytes()

# THIS IS THE ENCODED MASK FOR FE
encodedMask = base64.b64encode(im_bytes)
percentage = sky_ratio(mask)

# decodedMask = decode(encodedMask)
# cv2.imwrite('output/decoded_mask.jpg', decodedMask)

## determine % sky
# sky_mask = cv2.imread("output/mask.jpg")
# sky_ratio(decodedMask)