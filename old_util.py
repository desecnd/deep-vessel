import cv2
from pathlib import Path
from pydicom import dcmread

from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import pydicom._storage_sopclass_uids

from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity

from skimage.draw import line_nd

import numpy as np


def load_image_from_path(filepath: str):
    """ Load image in .dcm, .jpg or .JPG format. Return OpenCV data """
    path = Path(filepath)
    
    if not path.is_file:
        raise FileNotFoundError
        
    if path.suffix == '.dcm':
        ds = dcmread(path)
        # `arr` is a numpy.ndarray
        arr = ds.pixel_array
        return arr
    elif path.suffix in ['.jpg', '.JPG']:
        return cv2.imread(str(path), 0)
    else: 
        raise Exception

def save_image_as_dicom(filename: str, image, patient_data):
    img_converted = img_as_ubyte(rescale_intensity(image, out_range=(0.0, 1.0)))
    
    # Populate required values for file meta information
    meta = Dataset()
    meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian  

    ds = FileDataset(None, {}, preamble=b"\0" * 128)
    ds.file_meta = meta

    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    

    ds.PatientName = patient_data.get("PatientName")
    ds.PatientID = patient_data.get("PatientID")
    ds.ImageComments = patient_data.get("ImageComments")
    # format used by DICOM files is YYYYMMDD
    ds.InstanceCreationDate = patient_data.get("InstanceCreationDate")

    ds.Modality = "CT"
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.FrameOfReferenceUID = pydicom.uid.generate_uid()

    ds.BitsStored = 8
    ds.BitsAllocated = 8
    ds.SamplesPerPixel = 1
    ds.HighBit = 7

    ds.ImagesInAcquisition = 1
    ds.InstanceNumber = 1

    ds.Rows, ds.Columns = img_converted.shape

    ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"

    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0

    pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)

    ds.PixelData = img_converted.tobytes()

    ds.save_as(filename, write_like_original=False)

def positions(
    rows,
    cols,
    n_detectors,
    n_iters,
    angle_delta_rad,
    arc_len_rad
):
    radius = min(rows, cols) // 2 - 1
    center_x = cols / 2
    center_y = rows / 2

    circle_pos_rad = 0
    for i in range(n_iters):    
        # wyznaczamy na podstawie wzoru koordynaty
        x_emitter = radius * np.cos(circle_pos_rad)
        y_emitter = radius * np.sin(circle_pos_rad)

        # zaokrąglamy do dyskretnych i dodajemy aby przesunąc na środek obrazka
        col_emitter = round(x_emitter + center_x)
        row_emitter = round(y_emitter + center_y)
        
        detectors = []
        
        # na podstawie wzoru, dodajemy detektory
        for j in range(n_detectors):
            x_detector = radius * np.cos(circle_pos_rad + np.pi - arc_len_rad / 2 + j * arc_len_rad / (n_detectors - 1)) 
            y_detector = radius * np.sin(circle_pos_rad + np.pi - arc_len_rad / 2 + j * arc_len_rad / (n_detectors - 1)) 

            col_detector = round(x_detector + center_x)
            row_detector = round(y_detector + center_y)
            
            detectors.append((row_detector, col_detector))

        yield i, (row_emitter, col_emitter), detectors

        # zwiekszamy kat modulo 2pi
        circle_pos_rad += angle_delta_rad
        if circle_pos_rad > 2 * np.pi:
            circle_pos_rad -= 2 * np.pi
            
def create_sinogram(img, n_detectors, n_iters, angle_delta_rad, arc_len_rad):
    sinogram = np.zeros((n_iters, n_detectors))
    
    rows, cols = img.shape

    for view_id, emitter, detectors in positions(rows, cols, n_detectors=n_detectors, n_iters=n_iters, angle_delta_rad=angle_delta_rad, arc_len_rad=arc_len_rad):
        for detector_id, detector in enumerate(detectors):
            mask = line_nd(emitter, detector)

            sample = np.mean(img[mask])

            sinogram[view_id, detector_id] = sample 
            
    return sinogram

def make_kernel(func, size):
    #  Create kernel vector
    _kernel_func = np.vectorize(func, otypes=[float])
    kx = (size - 1) // 2
    return _kernel_func(range(-kx, kx + 1))

def filter_sinogram(sinogram, kernel, n_iters: int):
    for i in range(n_iters):
        sinogram[i, :] = np.convolve(sinogram[i, :], kernel, mode='same')
        
    return sinogram

def reconstruct(rows, cols, sinogram, n_detectors, n_iters, angle_delta_rad, arc_len_rad, save_history=True):
    out_img = np.zeros((rows, cols))

    out_history = None
    if save_history:
        out_history = np.zeros((n_iters, rows, cols))

    for view_id, emitter, detectors in positions(rows, cols, n_detectors, n_iters, angle_delta_rad, arc_len_rad):
        for detector_id, detector in enumerate(detectors):
            mask = line_nd(emitter, detector)

            out_img[mask] += sinogram[view_id, detector_id] 
            
            if save_history:
                out_history[view_id] = out_img 

    # normalize
    out_img = (out_img - np.min(out_img)) / np.ptp(out_img) * 255
    
    if save_history:
        out_history = (out_history - np.min(out_history)) / np.ptp(out_history) * 255

    return out_img, out_history


def make_video(frames, video_filename, resolution=768, fps=10):
    # make mp4 video of reconstruction process
    # YOU NEED FFMPEG FOR THIS TO WORK, I THINK

    n_frames = frames.shape[0]
    rows, cols = frames.shape[1], frames.shape[2]

    if rows > cols:
        vid_rows, vid_cols = resolution, int(resolution / rows * cols)
    else:
        vid_rows, vid_cols = int(resolution / cols * rows), resolution

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out_vid = cv2.VideoWriter(video_filename, fourcc, fps, (vid_rows, vid_cols))

    # TODO: change to enumerate? or something?
    for i in range(n_frames):
        frame = frames[i, :, :].astype('uint8')
        frame = cv2.resize(frame, (vid_rows, vid_cols))
        out_vid.write(frame)

    out_vid.release()

def calculate_mse(imgA, imgB):
    mask = np.zeros_like(imgA, dtype='float64')

    rows, cols = imgA.shape

    radius = min(rows, cols) // 2 - 1
    center_x = cols // 2
    center_y = rows // 2
    mask = cv2.circle(mask, (center_x, center_y), radius, color=1, thickness=-1)

    masked_se = np.square(imgA - imgB) * mask

    # i.e. mean square error
    mse = np.sum(masked_se) / np.sum(mask)

    return masked_se, mse