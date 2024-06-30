import os


def mimic_cxr_image_path(dir, subject_id, study_id, dicom_id, ext='dcm'):
    return os.path.join(dir, 'p' + str(subject_id)[:2], 'p' + str(subject_id),
                        's' + str(study_id), str(dicom_id) + '.' + ext)


def mimic_cxr_text_path(dir, subject_id, study_id, ext='txt'):
    return os.path.join(dir, 'p' + str(subject_id)[:2], 'p' + str(subject_id),
                        's' + str(study_id) + '.' + ext)
