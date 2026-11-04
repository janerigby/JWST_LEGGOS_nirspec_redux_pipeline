"""Fast standalone NIRSpec geometry masks for JWST 3.0 FFT flicker cleaning."""
import os
import json
from contextlib import ExitStack
from pathlib import Path

# STScI helpdesk says these os commands need to come BEFORE jwst pipeline packages are imported
home = "/Users/jrrigby1/Ref_files/"
os.environ["CRDS_PATH"] = home + "crds_cache/jwst_ops"
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
mycontext = 'jwst_1584.pmap'
os.environ["CRDS_CONTEXT"] =  mycontext   # v3.0.0

import numpy as np
from astropy.io import fits
import gwcs
from gwcs.utils import to_index
from jwst import datamodels
from jwst.assign_wcs import AssignWcsStep, nirspec
from jwst.msaflagopen import MSAFlagOpenStep
from stdatamodels.jwst.datamodels import dqflags, read_metadata


def make_custom_NIRSpec_mask(
    input_data,
    output_file,
    *,
    exp_type_override=None,
    mask_IFU=None,
    mask_MSA=None,
    mask_FS_strip=None,
    mask_FSprism=False,
    mask_failed_open=None,
    mask_box=False,
    mask_corners=None,
    mask_origin=1,
    single_mask=True,
    input_dir=None,
    overwrite=False,
):
    """Write geometry-only user_mask FITS; return True=background, False=exclude.

    Accept UNCAL/RATE/RATEINTS FITS paths or unextracted NIRSpec datamodels.
    FITS inputs are read for headers/metadata (and existing failed-open DQ
    flags), never SCI values. Build WCSs on small synthetic 2D ImageModels:
    no Detector1 calibration, reference-pixel correction, or ramp fitting.
    Raw IRS2 inputs use the science grid specified by SUBSIZE1/SUBSIZE2,
    checked against the raw shape; interleaved reference samples are not part
    of the output. Boxes always use RATE/science Image coordinates.

    None switches follow EXP_TYPE or exp_type_override: IFU on for NRS_IFU,
    MSA on for NRS_MSASPEC, failed-open flagging on for either. Existing
    MSA_FAILED_OPEN DQ flags are honored unless mask_failed_open=False.
    mask_MSA=True forces MSA geometry regardless of the header/override;
    it requires MSAMETFL, MSAMETID, PATT_NUM and the exposure's MSA file.
    input_dir defaults to the input FITS parent; it locates that MSA file.
    IFU and MSA exclusions use the JWST 3.0 mask helper algorithms. Failed
    shutters are projected by MSAFlagOpenStep using CRDS msaoper references.
    Missing metadata or references raises an error instead of guessing.

    mask_FS_strip covers every column in inclusive DS9 Image rows 950–1108.
    None enables it for IFU, or MOS without quadrant-5 fixed slits.
    mask_FSprism covers inclusive origin-1 corners (846, 952, 1571, 1105).
    mask_box adds an independent box (x1,y1,x2,y2), inclusive, with origin 1
    by default or mask_origin=0. Reversed/fractional corners are supported:
    exclude integer pixel centers inside the box, clipped to the image.
    These coordinates remain local to the science image for subarrays.

    No NaN/zero exclusions or image-statistics clipping are performed here.
    Use this mask with clean_flicker_noise fit_method='fft', autoparam=False,
    background_method='median' or 'model', n_sigma>0 to clip group differences
    in Detector1. background_method=None disables that additional clipping.

    single_mask=True writes a 2D mask for all integrations; False writes
    identical geometric planes (plus per-plane existing flags if supplied).
    Input values/metadata are unchanged, saved metadata retain the original
    EXP_TYPE, and existing outputs require overwrite=True.
    """
    output = Path(output_file).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    supported = {'NRS_IFU', 'NRS_MSASPEC', 'NRS_FIXEDSLIT', 'NRS_BRIGHTOBJ'}
    if exp_type_override is not None:
        if not isinstance(exp_type_override, str):
            raise TypeError('exp_type_override must be a string or None')
        exp_type_override = exp_type_override.strip().upper()
        if exp_type_override not in supported:
            raise ValueError(f'Unsupported exp_type_override: {exp_type_override!r}')
    for value in (mask_IFU, mask_MSA, mask_FS_strip, mask_failed_open):
        if value is not None and not isinstance(value, (bool, np.bool_)):
            raise TypeError('Mask switches must be True, False, or None')
    mask_failed_open = None if mask_failed_open is None else bool(mask_failed_open)
    for value in (mask_FSprism, mask_box, single_mask):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError('mask_FSprism, mask_box and single_mask must be boolean')
    boxes = [(845, 951, 1570, 1104)] if mask_FSprism else []
    if mask_box:
        if isinstance(mask_origin, (bool, np.bool_)) or mask_origin not in (0, 1):
            raise ValueError('mask_origin must be 0 or 1')
        corners = np.asarray(mask_corners, dtype=float)
        if corners.shape != (4,) or not np.all(np.isfinite(corners)):
            raise ValueError('mask_corners must contain four finite coordinates')
        x1, y1, x2, y2 = corners - mask_origin
        boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))

    with ExitStack() as stack:
        def own(model):
            stack.callback(model.close)
            return model

        if isinstance(input_data, (str, Path)):
            filename = Path(input_data).expanduser().resolve()
            if filename == output:
                raise ValueError('Output must differ from the science input')
            metadata = read_metadata(filename, flatten=False)
            image = own(datamodels.ImageModel())
            image.update({'meta': metadata['meta']})
            # No SCI access; non-memmapped DQ supports FITS unsigned BZERO.
            hdus = stack.enter_context(fits.open(filename, memmap=False))
            header = hdus['SCI'].header
            ndim = header['NAXIS']
            raw_shape = (header['NAXIS2'], header['NAXIS1'])
            nints = header.get('NAXIS4', 1) if ndim == 4 else header.get('NAXIS3', 1)
            directory = str(filename.parent) if input_dir is None else str(Path(input_dir).expanduser())
            existing_dq = None
            if mask_failed_open is not False:
                for name in ('DQ', 'PIXELDQ'):
                    if name in hdus:
                        existing_dq = hdus[name].data
                        break
        else:
            if not isinstance(input_data, (datamodels.ImageModel, datamodels.IFUImageModel,
                                           datamodels.CubeModel, datamodels.RampModel,
                                           datamodels.Level1bModel)):
                raise TypeError('Input must be an unextracted NIRSpec image, cube or ramp')
            image = own(datamodels.ImageModel())
            image.update(input_data)
            shape = input_data.data.shape
            ndim, raw_shape = len(shape), shape[-2:]
            nints = shape[0] if ndim > 2 else 1
            directory = '' if input_dir is None else str(Path(input_dir).expanduser())
            existing_dq = None
            if mask_failed_open is not False:
                existing_dq = getattr(input_data, 'dq', None)
                if existing_dq is None:
                    existing_dq = getattr(input_data, 'pixeldq', None)
        if ndim not in (2, 3, 4):
            raise ValueError('SCI must be a 2D image, 3D rateints cube, or 4D ramp')
        header_type = (image.meta.exposure.type or '').upper()
        selected = header_type if exp_type_override is None else exp_type_override
        if selected not in supported:
            raise ValueError(f'Unsupported NIRSpec EXP_TYPE: {selected!r}')
        detector = image.meta.instrument.detector
        if detector not in ('NRS1', 'NRS2'):
            raise ValueError('NIRSpec DETECTOR must be NRS1 or NRS2')

        science_shape = raw_shape
        if max(raw_shape) > 2048:
            # IRS2 DMS raw arrays include interleaved samples along rows.
            # Only metadata are needed to describe the post-refpix grid.
            science_shape = (image.meta.subarray.ysize, image.meta.subarray.xsize)
            if (ndim != 4 or 'IRS2' not in (image.meta.exposure.readpatt or '').upper()
                    or raw_shape != (3200, 2048) or science_shape != (2048, 2048)):
                raise ValueError('Unrecognized expanded raw grid; expected full-frame NIRSpec IRS2')
            if existing_dq is not None and existing_dq.shape[-2:] == raw_shape:
                from jwst.refpix.irs2_subtract_reference import make_irs2_mask
                normal = image.meta.exposure.nrs_normal
                reference = image.meta.exposure.nrs_reference
                rows = make_irs2_mask(3200, 2048, 16 if normal is None else normal,
                                     4 if reference is None else reference)
                if detector == 'NRS2':
                    rows = rows[::-1]
                existing_dq = existing_dq[..., rows, :]
        image.data = np.zeros(science_shape, dtype=np.float32)
        image.dq = np.zeros(science_shape, dtype=np.uint32)
        image.meta.wcs = None
        # Update sizes to describe the synthetic science image, not raw samples.
        image.meta.subarray.ysize, image.meta.subarray.xsize = science_shape
        use_ifu = selected == 'NRS_IFU' if mask_IFU is None else mask_IFU
        use_msa = selected == 'NRS_MSASPEC' if mask_MSA is None else mask_MSA
        models = {}

        def geometry(mode):
            if mode not in models:
                model = own(image.copy())
                model.meta.exposure.type = mode
                step = AssignWcsStep()
                step.input_dir = directory
                model = own(step.run(model))
                if getattr(model.meta, 'wcs', None) is None:
                    raise RuntimeError(f'WCS assignment did not produce a WCS for {mode}')
                models[mode] = model
            return models[mode]

        mask = np.ones(science_shape, dtype=bool)
        if use_ifu:
            # clean_flicker_noise.mask_ifu_slices algorithm, without its DQ map.
            for wcs in nirspec.nrs_ifu_wcs(geometry('NRS_IFU')):
                x, y = gwcs.wcstools.grid_from_bounding_box(wcs.bounding_box, step=(1, 1), center=True)
                valid = ~np.isnan(wcs(x, y)[2])
                mask[y[valid].astype(int), x[valid].astype(int)] = False
        if use_msa:
            # clean_flicker_noise.mask_slits algorithm on forced MOS WCS.
            model = geometry('NRS_MSASPEC')
            slits = model.meta.wcs.get_transform('gwa', 'slit_frame').slit_ids
            for slit in slits:
                bbox = model.meta.wcs.bounding_box[slit]
                xlo, xhi = to_index(bbox[0])
                ylo, yhi = to_index(bbox[1])
                mask[ylo:yhi, xlo:xhi] = False
        if mask_failed_open is True or (mask_failed_open is None and selected in ('NRS_IFU', 'NRS_MSASPEC')):
            step = MSAFlagOpenStep()
            step.input_dir = directory
            try:
                flagged = own(step.run(geometry(selected)))
            except json.JSONDecodeError as exc:
                reference = getattr(step, 'reference_name', None)
                size = 'unknown'
                if reference:
                    try:
                        size = Path(reference).stat().st_size
                    except OSError:
                        pass
                exc.add_note(
                    f'Failed-open shutter msaoper reference: {reference!r}; '
                    f'file size: {size} bytes. Validate this file as JSON; '
                    'if invalid, quarantine it and re-download it through CRDS '
                    'in a single process before restarting the worker pool.'
                )
                raise
            if flagged.meta.cal_step.msa_flagging != 'COMPLETE':
                raise RuntimeError('Failed-open shutter flagging did not complete')
            mask[(flagged.dq & dqflags.pixel['MSA_FAILED_OPEN']) != 0] = False
        strip = mask_FS_strip
        if strip is None:
            strip = selected == 'NRS_IFU'
            if selected == 'NRS_MSASPEC':
                slits = geometry('NRS_MSASPEC').meta.wcs.get_transform('gwa', 'slit_frame').slits
                strip = not any(s.quadrant == 5 for s in slits)
        if strip:
            mask[949:1108, :] = False
        height, width = science_shape
        for xlo, ylo, xhi, yhi in boxes:
            xstart, xstop = np.clip((np.ceil(xlo), np.floor(xhi) + 1), 0, width).astype(int)
            ystart, ystop = np.clip((np.ceil(ylo), np.floor(yhi) + 1), 0, height).astype(int)
            mask[ystart:ystop, xstart:xstop] = False
        if not single_mask:
            mask = np.broadcast_to(mask, (nints, height, width)).copy()
        if existing_dq is not None:
            if existing_dq.shape[-2:] != science_shape:
                raise ValueError('Existing DQ does not match the science grid')
            failed = (existing_dq & dqflags.pixel['MSA_FAILED_OPEN']) != 0
            if failed.ndim == 3 and single_mask:
                failed = np.any(failed, axis=0)
            if failed.ndim == 3 and not single_mask and failed.shape[0] != nints:
                raise ValueError('Existing DQ integration count does not match SCI')
            mask &= ~failed
        cls = datamodels.ImageModel if single_mask else datamodels.CubeModel
        saved = own(cls(data=mask.astype(np.float32)))
        saved.update(image)
        saved.save(str(output))
        return mask
