import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata, interp1d
from scipy.spatial import Delaunay
from scipy.ndimage import zoom, rotate
from scipy.interpolate import griddata


# Helper function to check header consistency
def check_header_consistency(headers):
    keys_to_check = [
        "OBSCURED",
        "PIXSCALE",
        "LAMBDA",
        "MINLAM",
        "MAXLAM",
        "D",
        "XCENTER",
        "YCENTER",
    ]
    for key in keys_to_check:
        values = [h.get(key) for h in headers if key in h]
        if len(set(values)) > 1:
            raise ValueError(f"Inconsistent {key} values in headers")


def load_coronagraph(
    Istarfile1,
    Istarfile2,
    angdiamsfile,
    PSFmapsfile,
    PSFoffsetsfile,
    skytransfile,
    photap_rad,
    PSF_trunc_ratio,
    nrolls=None,
    coro_bw_multiplier=None,
    coro_contrast_multiplier=None,
    coro_throughput_multiplier=None,
    coro_pixscale_multiplier=None,
    noisefloor_contrast=None,
    noisefloor_PPF=None,
    azavg_contrast=False,
):

    # Helper function to read FITS files and headers
    def read_fits(filename):
        with fits.open(filename, mode="readonly") as hdul:
            data = hdul[0].data
            header = hdul[0].header
        return data, header

    # Read all the input files
    Istar1, h1 = read_fits(Istarfile1)
    angdiams, h2 = read_fits(angdiamsfile)
    PSFmaps, h3 = read_fits(PSFmapsfile)
    PSFoffsets, h4 = read_fits(PSFoffsetsfile)
    skytrans, h5 = read_fits(skytransfile)

    # Check if Istarfile2 is provided
    if Istarfile2:
        Istar2, h8 = read_fits(Istarfile2)
    else:
        Istar2 = Istar1
        h8 = h1

    # Check consistency of header values
    headers = [h1, h2, h3, h4, h5, h8]
    check_header_consistency(headers)

    # Extract parameters from headers
    obscured = h1["OBSCURED"]
    pixscale = h1["PIXSCALE"]
    npix = Istar1.shape[0]  # Assuming square image
    lambd = h1["LAMBDA"]
    D = h1["D"]
    xcenter = h1["XCENTER"]
    ycenter = h1["YCENTER"]
    bw = (h1["MAXLAM"] - h1["MINLAM"]) / lambd

    # Set nrolls if not provided
    if nrolls is None or nrolls <= 0:
        nrolls = h1.get("NROLLS", 1)
    nrolls = max(nrolls, 1)

    # Modify performance if desired
    if coro_throughput_multiplier is not None and coro_throughput_multiplier != 1.0:
        print(
            f"Modifying coronagraph throughput by a factor of {coro_throughput_multiplier}"
        )
        Istar1 *= coro_throughput_multiplier
        Istar2 *= coro_throughput_multiplier
        PSFmaps *= coro_throughput_multiplier
        skytrans *= coro_throughput_multiplier

    if coro_contrast_multiplier is not None and coro_contrast_multiplier != 1.0:
        print(
            f"Modifying coronagraph contrast by a factor of {coro_contrast_multiplier}"
        )
        Istar1 *= coro_contrast_multiplier
        Istar2 *= coro_contrast_multiplier

    if coro_bw_multiplier is not None and coro_bw_multiplier != 1.0:
        print(f"Modifying coronagraph bandwidth by a factor of {coro_bw_multiplier}")
        bw *= coro_bw_multiplier
    if coro_pixscale_multiplier is not None and coro_pixscale_multiplier != 1.0:
        print(
            f"Modifying coronagraph pixscale by a factor of {coro_pixscale_multiplier}"
        )
        pixscale *= coro_pixscale_multiplier

    # Calculate Istar as the average of Istar1 and Istar2
    Istar = 0.5 * (Istar1 + Istar2)

    # Calculate noisefloor
    diff = Istar2 - Istar1
    ndiams = Istar1.shape[2]

    # Define the photometric aperture kernel
    kernel_size = int(2 * np.ceil(photap_rad / pixscale) + 1)
    y, x = np.ogrid[
        -kernel_size // 2 : kernel_size // 2 + 1,
        -kernel_size // 2 : kernel_size // 2 + 1,
    ]
    kernel = ((x * pixscale) ** 2 + (y * pixscale) ** 2 <= photap_rad**2).astype(float)
    kernel /= kernel.sum()

    # Convolve the difference with the kernel
    from scipy.signal import convolve2d

    for i in range(ndiams):
        diff[:, :, i] = convolve2d(diff[:, :, i], kernel, mode="same")

    # Calculate noisefloor
    y, x = np.ogrid[:npix, :npix]
    r = (
        np.sqrt((x - xcenter + npix / 2) ** 2 + (y - ycenter + npix / 2) ** 2)
        * pixscale
    )
    nvals = int(np.ceil((r.max() + 0.5 * pixscale * np.sqrt(2)) / pixscale))
    rvec = (np.arange(nvals) + 0.5) * pixscale

    noisefloor = np.zeros_like(Istar1)
    for i in range(nvals):
        mask1 = np.abs(r - rvec[i]) < photap_rad
        mask2 = np.abs(r - rvec[i]) < pixscale / 2
        if mask1.any() and mask2.any():
            for k in range(ndiams):
                stddev = np.std(diff[:, :, k][mask1])
                noisefloor[:, :, k] = np.maximum(noisefloor[:, :, k], mask2 * stddev)

    # Divide by the number of pixels in the photometric aperture
    omega = np.pi * photap_rad**2
    noisefloor /= omega / (pixscale**2)

    # Interpolate the offset PSF file to calculate photap_frac
    resolvingfactor = int(np.ceil(pixscale / 0.05))
    temppixomegalod = (pixscale / resolvingfactor) ** 2

    noffsets = PSFoffsets.shape[1]
    omega_lod = np.zeros((npix, npix, len(PSF_trunc_ratio)))
    photap_frac = np.zeros((npix, npix, len(PSF_trunc_ratio)))

    # Handle 1D and 2D offset cases
    if np.all(PSFoffsets[0, :] == PSFoffsets[0, 0]) or np.all(
        PSFoffsets[1, :] == PSFoffsets[1, 0]
    ):
        # 1D case
        offsets = np.sqrt(PSFoffsets[0, :] ** 2 + PSFoffsets[1, :] ** 2)
        peakvals = np.zeros(noffsets)
        temp_omega_lod = np.zeros((noffsets, len(PSF_trunc_ratio)))
        temp_photap_frac = np.zeros((noffsets, len(PSF_trunc_ratio)))

        y, x = np.ogrid[:npix, :npix]
        r = (
            np.sqrt((x - xcenter + npix / 2) ** 2 + (y - ycenter + npix / 2) ** 2)
            * pixscale
        )
        resolvedPSFs = np.maximum(
            0, zoom(PSFmaps, (resolvingfactor, resolvingfactor, 1))
        )

        for i in range(noffsets):
            tempPSF = PSFmaps[:, :, i]
            norm = np.sum(tempPSF)
            peakvals[i] = np.max(tempPSF)

            tempPSF = resolvedPSFs[:, :, i]
            norm2 = np.sum(tempPSF)
            tempPSF *= norm / norm2
            maxtempPSF = np.max(tempPSF)

            for j, ratio in enumerate(PSF_trunc_ratio):
                mask = tempPSF > ratio * maxtempPSF
                temp_omega_lod[i, j] = np.sum(mask) * temppixomegalod
                temp_photap_frac[i, j] = np.sum(tempPSF[mask])

        PSFpeaks = np.interp(r, offsets, peakvals)
        for j in range(len(PSF_trunc_ratio)):
            omega_lod[:, :, j] = np.interp(r, offsets, temp_omega_lod[:, j])
            photap_frac[:, :, j] = np.interp(r, offsets, temp_photap_frac[:, j])

    else:
        # 2D case
        peakvals = np.zeros(noffsets)
        temp_omega_lod = np.zeros((noffsets, len(PSF_trunc_ratio)))
        temp_photap_frac = np.zeros((noffsets, len(PSF_trunc_ratio)))

        resolvedPSFs = np.maximum(
            0, zoom(PSFmaps, (resolvingfactor, resolvingfactor, 1))
        )

        for i in range(noffsets):
            tempPSF = PSFmaps[:, :, i]
            norm = np.sum(tempPSF)
            peakvals[i] = np.max(tempPSF)

            tempPSF = resolvedPSFs[:, :, i]
            norm2 = np.sum(tempPSF)
            tempPSF *= norm / norm2
            maxtempPSF = np.max(tempPSF)

            for j, ratio in enumerate(PSF_trunc_ratio):
                mask = tempPSF > ratio * maxtempPSF
                temp_omega_lod[i, j] = np.sum(mask) * temppixomegalod
                temp_photap_frac[i, j] = np.sum(tempPSF[mask])

        if np.min(PSFoffsets[0, :]) > 0 and np.min(PSFoffsets[1, :]) > 0:
            # PSF offsets are only for positive quadrant
            tempPSFoffsetsx = np.tile(PSFoffsets[0, :], 4)
            tempPSFoffsetsy = np.tile(PSFoffsets[1, :], 4)
            tempPSFoffsetsx[noffsets : 2 * noffsets] *= -1
            tempPSFoffsetsx[2 * noffsets : 3 * noffsets] *= -1
            tempPSFoffsetsy[2 * noffsets :] *= -1

            temppeakvals = np.tile(peakvals, 4)

            points = np.column_stack((tempPSFoffsetsx, tempPSFoffsetsy))
            grid_x, grid_y = np.mgrid[
                -npix * pixscale / 2 : npix * pixscale / 2 : npix * 1j,
                -npix * pixscale / 2 : npix * pixscale / 2 : npix * 1j,
            ]

            PSFpeaks = griddata(points, temppeakvals, (grid_x, grid_y), method="linear")

            for j in range(len(PSF_trunc_ratio)):
                junk = np.tile(temp_omega_lod[:, j], 4)
                omega_lod[:, :, j] = griddata(
                    points, junk, (grid_x, grid_y), method="linear"
                )

                junk = np.tile(temp_photap_frac[:, j], 4)
                photap_frac[:, :, j] = griddata(
                    points, junk, (grid_x, grid_y), method="linear"
                )

        else:
            points = PSFoffsets.T
            grid_x, grid_y = np.mgrid[
                -npix * pixscale / 2 : npix * pixscale / 2 : npix * 1j,
                -npix * pixscale / 2 : npix * pixscale / 2 : npix * 1j,
            ]

            PSFpeaks = griddata(points, peakvals, (grid_x, grid_y), method="linear")

            for j in range(len(PSF_trunc_ratio)):
                omega_lod[:, :, j] = griddata(
                    points, temp_omega_lod[:, j], (grid_x, grid_y), method="linear"
                )
                photap_frac[:, :, j] = griddata(
                    points, temp_photap_frac[:, j], (grid_x, grid_y), method="linear"
                )

        omega_lod = np.maximum(omega_lod, 0)
        photap_frac = np.maximum(photap_frac, 0)

        k = np.where(np.min(Istar, axis=2) == 0)
        if len(k[0]) > 0:
            for j in range(len(PSF_trunc_ratio)):
                temp_photap_frac = photap_frac[:, :, j]
                temp_photap_frac[k] = 0
                photap_frac[:, :, j] = temp_photap_frac

    # Azimuthally average the leaked starlight maps and noise floor if requested
    if azavg_contrast:
        print("Azimuthally averaging contrast maps and noise floor...")
        ntheta = 100
        theta = np.linspace(0, 360, ntheta)
        tempIstar = np.zeros_like(Istar)
        tempnoisefloor = np.zeros_like(noisefloor)

        for itheta in range(ntheta):
            for i in range(Istar.shape[2]):
                rotated_Istar = rotate(
                    Istar[:, :, i],
                    theta[itheta],
                    reshape=False,
                    center=(xcenter - 0.5, ycenter - 0.5),
                )
                tempIstar[:, :, i] += rotated_Istar

                rotated_noisefloor = rotate(
                    noisefloor[:, :, i],
                    theta[itheta],
                    reshape=False,
                    center=(xcenter - 0.5, ycenter - 0.5),
                )
                tempnoisefloor[:, :, i] += rotated_noisefloor

        Istar = tempIstar / ntheta
        noisefloor = tempnoisefloor / ntheta

    # Calculate contrast
    contrast = np.zeros_like(Istar)
    for i in range(Istar.shape[2]):
        contrast[:, :, i] = Istar[:, :, i] / PSFpeaks

    skytrans = np.maximum(skytrans, 0)
    # Set the noise floor map if noisefloor_contrast or noisefloor_PPF is provided
    if noisefloor_contrast is not None:
        print("Setting the noise floor via user-supplied noisefloor_contrast...")
        j = np.argmin(np.abs(np.array(PSF_trunc_ratio) - 0.3))
        for i in range(Istar.shape[2]):
            noisefloor[:, :, i] = (
                (pixscale**2 / omega_lod[:, :, j])
                * noisefloor_contrast
                * photap_frac[:, :, j]
            )

    if noisefloor_PPF is not None:
        print("Setting the noise floor via user-supplied noisefloor_PPF...")
        noisefloor = Istar / noisefloor_PPF

    cfloor = np.zeros_like(Istar)
    for i in range(Istar.shape[2]):
        cfloor[:, :, i] = noisefloor[:, :, i] / PSFpeaks

    # Return all the calculated and processed values
    return {
        "Istar": Istar,
        "noisefloor": noisefloor,
        "photap_frac": photap_frac,
        "omega_lod": omega_lod,
        "skytrans": skytrans,
        "pixscale": pixscale,
        "npix": npix,
        "xcenter": xcenter,
        "ycenter": ycenter,
        "bw": bw,
        "angdiams": angdiams,
        "ndiams": angdiams.shape[0],
        "npsfratios": len(PSF_trunc_ratio),
        "nrolls": nrolls,
        "psf_trunc_ratio": np.array(PSF_trunc_ratio),
        "contrast": contrast,
        "cfloor": cfloor,
        "PSFpeaks": PSFpeaks,
        "D": D,
        "lambd": lambd,
        "obscured": obscured,
    }


def validate_outputs(output_dict, input_dict=None):
    """
    Print validation information for the load_coronagraph function.

    :param output_dict: Dictionary of outputs from load_coronagraph
    :param input_dict: Dictionary of inputs to load_coronagraph (optional)
    """

    def print_array_info(name, arr):
        print(f"{name} shape: {arr.shape}")
        print(
            f"{name} min, max, mean: {arr.min():.6e}, {arr.max():.6e}, {arr.mean():.6e}"
        )
        print(f"{name} center element: {arr[arr.shape[0]//2, arr.shape[1]//2]:.6e}")
        print(f"{name} corner elements: {arr[0,0]:.6e}, {arr[-1,-1]:.6e}")
        print()

    if input_dict:
        print("Input Validation:")
        for key in ["Istar1", "PSFmaps", "PSFoffsets", "skytrans"]:
            if key in input_dict:
                print_array_info(key, input_dict[key])

        print("Header Values:")
        for key in ["pixscale", "npix", "lambd", "D", "xcenter", "ycenter", "bw"]:
            if key in input_dict:
                print(f"{key}: {input_dict[key]}")
        print()

    print("Output Validation:")
    for key in [
        "Istar",
        "noisefloor",
        "omega_lod",
        "photap_frac",
        "contrast",
        "cfloor",
        "PSFpeaks",
    ]:
        if key in output_dict:
            print_array_info(key, output_dict[key])

    print("Scalar Outputs:")
    for key in [
        "pixscale",
        "npix",
        "xcenter",
        "ycenter",
        "bw",
        "ndiams",
        "npsfratios",
        "nrolls",
    ]:
        if key in output_dict:
            print(f"{key}: {output_dict[key]}")


import os
import glob
from astropy.io import fits
import numpy as np


def file_search(directory, filename):
    """Mimics IDL's file_search function"""
    return glob.glob(os.path.join(directory, filename))


corodir1 = "/Users/ealei/Downloads/usort_offaxis_ovc/"
Istarfile1_1 = file_search(corodir1, "stellar_intens.fits")[0]
Istarfile1_2 = file_search(corodir1, "stellar_intens.fits")[0]
angdiamsfile1 = file_search(corodir1, "stellar_intens_diam_list.fits")[0]
PSFmapsfile1 = file_search(corodir1, "offax_psf.fits")[0]
PSFoffsetsfile1 = file_search(corodir1, "offax_psf_offset_list.fits")[0]
skytransfile1 = file_search(corodir1, "sky_trans.fits")[0]
python_output = load_coronagraph(
    Istarfile1_1,
    Istarfile1_2,
    angdiamsfile1,
    PSFmapsfile1,
    PSFoffsetsfile1,
    skytransfile1,
    photap_rad=0.85,
    PSF_trunc_ratio=[0.3],
    noisefloor_PPF=0.03,
)


# Run your Python function
# Print validation info
validate_outputs(python_output)
