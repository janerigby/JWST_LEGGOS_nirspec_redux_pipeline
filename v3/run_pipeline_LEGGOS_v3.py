# Script to reduce LEGGOS JWST NIRSpec IFU data.
# This was a script by Brian Welch, running v1.20 of the pipeline.
# Jane Rigby upgraded to run v2 and v3 of the pipeline, to check for missing CRDS files,
# and to run in parallel.  Codex 5.6 was used as a coding assistant

import numpy as np
import glob
import os
import subprocess
import sys
import warnings
import jwst
from copy import deepcopy
from multiprocessing import current_process
from multiprocessing.pool import Pool
from packaging.version import InvalidVersion, Version

###############################################
# Directories to customize for your machine
dir_calib = "/Users/jrrigby1/Ref_files/"
dir_inputdata  = "/Volumes/Rawdata/Raw_LEGGOS/PID04125/"
dir_outputdata =  "/Users/jrrigby1/SCIENCE/JWST_Data/LEGGOS/Reduced_v3.0/PID04125/"
dir_bkgfiles = "/Users/jrrigby1/SCIENCE/JWST_Data/LEGGOS/Backgrounds/"


pipev = jwst.__version__

try:
    pipeline_version = Version(pipev)
except InvalidVersion as error:
    raise RuntimeError(
        f'Cannot interpret JWST pipeline version {pipev!r}; stopping.'
    ) from error

# The clean_flicker_noise least-squares calculation can overflow internally
# while handling masked/invalid pixels. Suppress only these known warnings;
# unrelated RuntimeWarnings remain visible.
warnings.filterwarnings(
    "ignore",
    message=r"(?:overflow encountered|invalid value encountered) in matmul",
    category=RuntimeWarning,
    module=r"jwst\.clean_flicker_noise\.lib",
)


if pipeline_version == Version('1.20.2'):
    det1_param_nested_dict = {
        'save_results': True,
        'jump': {'maximum_cores': 'half', 'expand_large_events': True},
        'ramp_fit': {'maximum_cores': 'half'},
        'clean_flicker_noise': {'skip': False, 'fit_method': 'fft',
            'mask_science_regions': True},}
    spec2_param_nested_dict = {
        'save_results': True,
        'srctype': {'source_type': 'EXTENDED'},
        'cube_build': {'skip': True},
        'extract_1d': {'skip': True},
        'pixel_replace': {'skip': False, 'algorithm': 'mingrad'},}
    spec3_param_nested_dict = {
        'save_results': True,
        'outlier_detection': {'skip': False, 'ifu_second_check': True},
        'cube_build': {'coord_system': 'skyalign'},
        'extract_1d': {'skip': True},}

elif pipeline_version.major == 2:  # Settings introduced for v2.0.1
    det1_param_nested_dict = {
        'save_results': True,
        'jump': {'expand_large_events': True},
        'picture_frame': {'skip': False},
        'clean_flicker_noise': {'skip': False, 'autoparam': False,
            'fit_method': 'fft', 'background_method': None, 'n_sigma': 3, \
            'mask_science_regions': True, 'save_noise': False},}
    spec2_param_nested_dict = {
        'save_results': True,
        'srctype': {'source_type': 'EXTENDED'},
        'cube_build': {'skip': True},
        'extract_1d': {'skip': True},
        'pixel_replace': {'skip': False, 'algorithm': 'mingrad'},}
    spec3_param_nested_dict = {
        'save_results': True,
        'outlier_detection': {'skip': False, 'ifu_second_check': True},
        'cube_build': {'coord_system': 'skyalign'},
        'extract_1d': {'skip': True},
        'adaptive_trace_model': {'skip': False, 'maximum_cores': 'half', 'oversample': 3.0},
    }

elif pipeline_version.major == 3:
    # Initial v3 configuration: v2 settings plus documented v3 additions.
    det1_param_nested_dict = {
        'save_results': True,
        'jump': {'expand_large_events': True},
        'picture_frame': {'skip': False},
        'clean_flicker_noise': {'skip': False, 'autoparam': False,
            'fit_method': 'fft', 'background_method': None, 'n_sigma': 3, \
            'mask_science_regions': True, 'save_noise': False},}
    spec2_param_nested_dict = {
        'save_results': True,
        'srctype': {'source_type': 'EXTENDED'},
        'cube_build': {'skip': True},
        'extract_1d': {'skip': True},
        'pixel_replace': {'skip': False, 'algorithm': 'mingrad'},}
    spec3_param_nested_dict = {
        'save_results': True,
        'outlier_detection': {'skip': False, 'ifu_second_check': True},
        'cube_build': {'coord_system': 'skyalign', 'linear_wave': False},
        'extract_1d': {'skip': True},
        'adaptive_trace_model': {'skip': False, 'maximum_cores': 'half', 'oversample': 3.0, 'save_intermediate_results': True},
    }

else:
    raise RuntimeError(
        f'Unexpected JWST pipeline version {pipev}; supported configurations '
        'are 1.20.2, 2.x, and explicitly configured 3.x releases. Stopping.'
    )


def apply_pipeline_parameters(pipeline, parameters):
    """Recursively apply a nested parameter dictionary to a pipeline."""
    for name, value in parameters.items():
        if isinstance(value, dict):
            apply_pipeline_parameters(getattr(pipeline, name), value)
        else:
            setattr(pipeline, name, value)


_crds_preflight_context = None


def preflight_crds(input_files, stage_name):
    """Cache CRDS mappings and input-specific best references before a pool."""
    global _crds_preflight_context
    if _crds_preflight_context is None:
        import crds
        from crds.client import api as crds_api

        context = crds.get_default_context('jwst')
        print(f'CRDS preflight: caching mapping closure for {context}')
        crds_api.dump_mappings(context)

        # Spawned workers inherit the automatically selected literal context
        # and do not repeat context resolution or race to cache mappings.
        os.environ['CRDS_CONTEXT'] = context
        _crds_preflight_context = context
        print(f'CRDS mapping preflight complete: {context}')

    context = _crds_preflight_context
    fits_files = sorted({
        os.path.abspath(filename) for filename in input_files
        if str(filename).lower().endswith(('.fits', '.fit'))
    })
    if not fits_files:
        print(f'CRDS bestrefs preflight: no FITS inputs for {stage_name}')
        return context

    crds_executable = os.path.join(os.path.dirname(sys.executable), 'crds')
    command = [
        crds_executable, 'bestrefs',
        '--new-context', context,
        '--sync-mappings', '1',
        '--sync-references', '1',
        '--files', *fits_files,
    ]
    print(
        f'CRDS bestrefs preflight: checking {len(fits_files)} '
        f'{stage_name} input file(s).\n'
        'CRDS will now download any missing reference files. Some NIRSpec '
        'references are several GB, so this may take a while and may appear '
        'inactive. The worker pool will start only after all downloads finish.',
        flush=True,
    )
    subprocess.run(command, check=True)
    print(
        f'CRDS bestrefs preflight complete for {stage_name}; starting workers.',
        flush=True,
    )
    return context


def check_pipeline_step_statuses(result, input_file):
    """Raise if any returned data model records a pipeline step as FAILED."""
    if result is None:
        return
    models = [result] if hasattr(result, 'meta') else result
    failed_steps = set()
    for model in models:
        cal_step = getattr(getattr(model, 'meta', None), 'cal_step', None)
        statuses = getattr(cal_step, '_instance', {})
        failed_steps.update(
            step_name for step_name, status in statuses.items()
            if status == 'FAILED'
        )
    if failed_steps:
        failed_list = ', '.join(sorted(failed_steps))
        raise RuntimeError(
            f'Pipeline step(s) reported FAILED for {input_file}: {failed_list}'
        )


def close_pipeline_result(result):
    """Close a returned data model or model container when possible."""
    close = getattr(result, 'close', None)
    if close is not None:
        close()


def run_jwst_det1(input_file, parameters):
    """Run Detector1 for one exposure inside a worker process."""
    det1 = Detector1Pipeline()
    apply_pipeline_parameters(det1, parameters)
    result = det1.run(input_file)
    try:
        check_pipeline_step_statuses(result, input_file)
    finally:
        close_pipeline_result(result)
    return input_file


def run_jwst_spec2(input_file, parameters):
    """Run Spec2 for one rate file or association inside a worker process."""
    spec2 = Spec2Pipeline()
    apply_pipeline_parameters(spec2, parameters)
    result = spec2.run(input_file)
    try:
        check_pipeline_step_statuses(result, input_file)
    finally:
        close_pipeline_result(result)
    return input_file


def run_jwst_spec3(spec3, input_file):
    """Run Spec3 sequentially and reject any FAILED step status."""
    result = spec3.run(input_file)
    try:
        check_pipeline_step_statuses(result, input_file)
    finally:
        close_pipeline_result(result)


def run_pipeline_tasks(worker, tasks, cores2use):
    """Run prepared ``(input_file, parameters)`` tasks sequentially or in a pool."""
    if not tasks:
        return []
    if not isinstance(cores2use, int) or isinstance(cores2use, bool) or cores2use < 1:
        raise ValueError(f'cores2use must be a positive integer; got {cores2use!r}')
    if cores2use == 1:
        return [worker(*task) for task in tasks]

    worker_count = min(cores2use, len(tasks))
    pool = Pool(worker_count)
    try:
        results = pool.starmap(worker, tasks)
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()
        return results



###############################################
# Set up CRDS path and server environment variables
os.environ["CRDS_PATH"] = os.path.join(dir_calib, "crds_cache/jwst_ops")
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
# Let CRDS select the build context associated with the installed JWST version.
# Only the parent clears a possibly stale shell setting. Pool workers retain
# the literal context selected dynamically by preflight_crds().
if current_process().name == 'MainProcess':
    os.environ.pop("CRDS_CONTEXT", None)

import zipfile
import urllib.request

import json
import asdf

from astropy.io import fits
from astropy.utils.data import download_file
import astropy.units as u
from astropy import wcs
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, ManualInterval, LogStretch, LinearStretch, AsinhStretch

import matplotlib.pyplot as plt
import matplotlib as mpl

# The calwebb_spec and spec3 pipelines
from jwst.pipeline import Spec2Pipeline
from jwst.pipeline import Spec3Pipeline

# the level1 pipeline:
from jwst.pipeline import Detector1Pipeline

# data models
from jwst import datamodels

# association file utilities
from jwst.associations import asn_from_list as afl # Tools for creating association files
from jwst.associations.lib.rules_level2_base import DMSLevel2bBase # Definition of a Lvl2 association file
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base # Definition of a Lvl3 association file

#from stcal import dqflags

from scipy.interpolate import interp1d

targetlist = ['SGAS1110','SGAS2111','SGAS1050','SGAS1527','SGAS1429','COSMIC.EYE',
                'SGAS1723','SGAS1226','SPT0418-47','SPT2147-50', 'SGAS1402m28']

def run_pipeline(target, rundet1=True, runspec2=True, runspec3=True, runspec3_PRISM=False,
                 clean_data=True, run_bgsub=True, cube_resolutions=(0.1, 0.05),
                 cores2use=4):
    '''
    runspec3_PRISM == use non-linear wavelength solution for prism observations
        only runs the prism data through spec3 pipeline (things break trying to do all data here)
    clean_data = whether or not to run the "cleaning" step before stage 3 (turn off to save time when only running spec3)
    run_bgsub = whether or not to subtract the expected background from completed cubes
    cube_resolutions = spatial pixel scales to build in the Spec3 pipeline
    cores2use = number of worker processes for Detector1 and Spec2; use 1 for sequential execution
    '''
    if not isinstance(cores2use, int) or isinstance(cores2use, bool) or cores2use < 1:
        raise ValueError(f'cores2use must be a positive integer; got {cores2use!r}')
    print(f"JWST pipeline version: {pipev}")
    print(f'Starting NIRSpec IFU pipeline for {target}')
    # The input and output directory names are the same as the target name.
    target_directories = [
        target_name for target_name in targetlist + ['SPARKLER', 'SUNBURST', 'J1757132']
        if glob.glob(os.path.join(dir_inputdata, target_name, '**', '*_uncal.fits'), recursive=True)
    ]

    for target_name in target_directories:
        if target == target_name:
            this_inputdir = os.path.join(dir_inputdata, target_name, '')
            this_outputdir = os.path.join(dir_outputdata, target_name, '')
            break
    else:
        raise ValueError(f'No input directory containing data is available for {target}')

    input_path = this_inputdir
    output_path = this_outputdir


    if os.path.exists(output_path) == False: # if folder doesn't exist
        print('Creating folder ' + output_path)
        os.makedirs(output_path, exist_ok=True) # creates the folder and any missing parents


    # test by TAH to see if the files are accessible

    files = glob.glob(input_path + '**/*_uncal.fits', recursive=True) #list the uncalibrated (level 1b) files.
    files = sorted(files)

    ifu,sky = 0,0

    for exposure in files:
        with fits.open(exposure) as test:
            head = test[0].header
        if head['BKGDTARG'] == False: ifu += 1
        elif head['BKGDTARG'] == True: sky += 1

    print(f'Number of IFU: {ifu}, Number of sky: {sky}')


    # checking that the file system is in place for these data
    # if not, creating the folders
    folders_L2a = ['L2a/','L2a/sci/','L2a/bkg/']

    for folder in folders_L2a:
        if os.path.exists(output_path + folder) == False: # if folder doesn't exist
            print('Creating folder ' + output_path + folder)
            os.makedirs(output_path + folder, exist_ok=True) # creates the folder


    # Run the pipeline, splitting outputs into "sci" and "bkg" output folders
    # Leakcals will be in the same folders as their associated observations
    if rundet1 == True:
        print(f'Preparing Detector1 Pipeline tasks for {target}')
        det1_tasks = []
        for exposure in files:
            # set output directory based on sci vs bkg exposure
            with fits.open(exposure) as hdu:
                head = hdu[0].header.copy()
            det1_params = deepcopy(det1_param_nested_dict)
            if head['BKGDTARG'] == False:
                det1_params['output_dir'] = output_path + folders_L2a[1]
            elif head['BKGDTARG'] == True:
                det1_params['output_dir'] = output_path + folders_L2a[2]
            else: print('not target')
            if head['IS_IMPRT'] == True:
                print('Skipping leak cal file: ',exposure)
                continue
            # NSClean step moved to Detector1 pipeline
            # throws an error if there's nothing on NRS2, so
            if 'M' in head['GRATING'] and 'nrs2' in exposure:
                det1_params['clean_flicker_noise']['skip'] = True

            # Each exposure already occupies a pool worker. Avoid starting a
            # second multiprocessing layer inside the JWST steps.
            if cores2use > 1:
                for step_name in ('jump', 'ramp_fit'):
                    step_params = det1_params.get(step_name, {})
                    if 'maximum_cores' in step_params:
                        step_params['maximum_cores'] = '1'
            det1_tasks.append((exposure, det1_params))

        print(
            f'Running {len(det1_tasks)} Detector1 task(s) with '
            f'{min(cores2use, len(det1_tasks)) if det1_tasks else 0} worker(s)'
        )
        preflight_crds(
            [input_file for input_file, _parameters in det1_tasks],
            'Detector1',
        )
        run_pipeline_tasks(run_jwst_det1, det1_tasks, cores2use)

        print(f'Done Detector1 for {target}')

    # checking that the file system is in place for these data
    # if not, creating the folders
    folders_L2b = ['L2b/','L2b/sci/','L2b/bkg/']

    for folder in folders_L2b:
        if os.path.exists(output_path + folder) == False: # if folder doesn't exist
            print('Creating folder ' + output_path + folder)
            os.makedirs(output_path + folder, exist_ok=True) # creates the folder

    # define some path names for easy access
    use_leak = False
    process_background = False

    level1_sci_dir = os.path.join(output_path, folders_L2a[1]) 
    level2outputdir = os.path.join(output_path, folders_L2b[0])

    ratefiles = sorted(glob.glob(os.path.join(level1_sci_dir,'*rate.fits')))

    # And now we run the pipeline with the asn files we just made! 
    if runspec2 == True:
        print(f'Preparing Spec2 Pipeline tasks for {target}')
        spec2_tasks = []
        # this first part is kinda not needed, but its still here in case we want to reduce the background files at some point
        if (use_leak == True) or (process_background == True):
            if target == 'SGAS1723':
                asnfiles_sci = glob.glob(level2outputdir+'jw'+program_id+'-o'+obs_num_sci[0]+'*asn.json') 
                asnfiles_sci += glob.glob(level2outputdir+'jw'+program_id+'-o'+obs_num_sci[1]+'*asn.json')
            else:
                asnfiles_sci = glob.glob(level2outputdir+'jw'+'*asn.json')
            
            for asn in asnfiles_sci:
                #print(asn)
                spec2_params = deepcopy(spec2_param_nested_dict)
                spec2_params['output_dir'] = output_path + folders_L2b[1]
                # These steps were only explicitly skipped for direct rate-file runs.
                spec2_params.pop('cube_build', None)
                spec2_params.pop('extract_1d', None)
                # NSClean in Spec2 is deprecated, moved to Detector1 pipeline 
                #spec2.nsclean.skip = False
                #spec2.nsclean.save_mask = True
                spec2_tasks.append((asn, spec2_params))
            
            if process_background == True:
                asnfiles_bkg = glob.glob(level2outputdir+'jw'+program_id+'-o'+obs_num_bkg+'*asn.json')
                for asn in asnfiles_bkg:
                    spec2_params = deepcopy(spec2_param_nested_dict)
                    spec2_params['output_dir'] = output_path + folders_L2b[2]
                    # Preserve the original background-processing defaults.
                    spec2_params.pop('cube_build', None)
                    spec2_params.pop('extract_1d', None)
                    spec2_params.pop('pixel_replace', None)
                    spec2_tasks.append((asn, spec2_params))

        else:
            for file in ratefiles:
                with fits.open(file) as hdu:
                    head = hdu[0].header.copy()
                if 'M' in head['GRATING']: 
                    if 'nrs2' in file: continue
                    #print(file)
                    spec2_params = deepcopy(spec2_param_nested_dict)
                    spec2_params['output_dir'] = output_path + folders_L2b[1]
                    # NSClean in Spec2 is deprecated, moved to Detector1 pipeline 
                    #spec2.nsclean.skip = False
                    #spec2.nsclean.save_mask = True
                    spec2_tasks.append((file, spec2_params))

        print(
            f'Running {len(spec2_tasks)} Spec2 task(s) with '
            f'{min(cores2use, len(spec2_tasks)) if spec2_tasks else 0} worker(s)'
        )
        preflight_crds(
            [input_file for input_file, _parameters in spec2_tasks],
            'Spec2',
        )
        run_pipeline_tasks(run_jwst_spec2, spec2_tasks, cores2use)
        print(f'Done Spec2 Pipeline for {target}')


    # checking that the file system is in place for these data
    # if not, creating the folders
    folders_L3 = ['L3/']#, 

    for folder in folders_L3:
        if os.path.exists(output_path + folder) == False: # if folder doesn't exist
            print('Creating folder ' + output_path + folder)
            os.makedirs(output_path + folder, exist_ok=True) # creates the folder


    #orig_calfiles = glob.glob(level2outputdir + '/sci/*cal.fits') # cal files output from L2 pipeline

    # set different thresholds for each target - done by-eye w/ L2 ratefiles
    calmax_by_target = {
        'SGAS1723': 2000,
        'SGAS1110': 500,  # for g140h grating
        'SGAS2111': 50,
        'SGAS1050': 1000,  # increased from 500 on July 14 2026
        'SGAS1429': 400,
        'SGAS1527': 300,  # increased from 150 on July 14 2026
        'COSMIC.EYE': 200,
        'SGAS1226': 50,
        'SPT0418-47': 40,
        'SPT2147-50': 30,
        'SPARKLER': 15,
    }
    calmax = calmax_by_target.get(target)

    calfiles = glob.glob(level2outputdir + '/sci/*cal.fits') # cal files output from L2 pipeline

    #for file in calfiles:
    #   outfile = file[:-5] + '2.fits'
    #   if target == 'SUNBURST':
    #       if ('_02101_' in file) or ('_10101_' in file):
    #           cut_cal(file, outfile, max_threshold=3000, min_threshold=-10)
    #       elif ('_04101_' in file) or ('_12101_' in file):
    #           cut_cal(file, outfile, max_threshold=1800, min_threshold=-10)
    #       elif ('_06101_' in file) or ('_14101_' in file):
    #           cut_cal(file, outfile, max_threshold=2300, min_threshold=-10)
    #       else:
    #           cut_cal(file, outfile, max_threshold=2000, min_threshold=-10)
    #   else:
    #       cut_cal(file, outfile, max_threshold=calmax, min_threshold=-10)
    #print(outfile)

    if process_background == True:
        bkg_calfiles = glob.glob(level2outputdir + '/bkg/*cal.fits') # cal files output from L2 pipeline
        for file in bkg_calfiles:
            outfile = file[:-5] + '2.fits'
            cut_cal(file, outfile, max_threshold=calmax, min_threshold=-10)
            #print(outfile)

    input_spec2 = output_path + folders_L2b[1]
    clean_dir = output_path + folders_L2b[1] # just chuck them all in the same directory. Fuck me up fam. 
    #clean_data = True # turn to True to flag additional pixels - False will not set additional DQ flags
    if clean_data:
        for file in glob.glob(input_spec2+'*cal.fits'):
            run_clean_step(file, clean_dir, expand_flat=False)

    if target == 'COSMIC.EYE':
        target = 'COSMICEYE' # for better saving purposes

    # Make association files 
    calfiles = glob.glob(output_path + folders_L2b[1] + '*cal.fits')

    #bkgfiles = glob.glob(output_path + folders_L2b[2] + '*x1d.fits')
    if process_background == True:
        bkgfiles_cal = glob.glob(output_path + folders_L2b[2] + '*cal.fits') # for background testing only

    version = 'nobg' 
    #if process_background == True:
    #    version = 'standard'


    if version == 'standard':
        if len(folders_L3) == 1:
            fold = folders_L3[0]
            asnfile = os.path.join(output_path, fold, 'L3asn.json')
        else:
            fold = folders_L3[1]
            asnfile = os.path.join(output_path, fold, 'L3asn.json')
            lev3asnname = 'Level3_' + target + '_ALLOBS'
            writel3asn(calfiles, bkgfiles, asnfile, lev3asnname) 
        
    if version == 'bgonly':
        fold = folders_L3[0]
        asnfile = os.path.join(output_path, fold, 'L3asn.json')
        lev3asnname = 'Level3_' + target + '_BGONLY'
        writel3asn(bkgfiles_cal, None, asnfile, lev3asnname)

    if version == 'nobg':
        if len(folders_L3) == 1:
            fold = folders_L3[0]
            asnfile = os.path.join(output_path, fold, 'L3asn.json')
        else:
            fold = folders_L3[3]
            asnfile = os.path.join(output_path, fold, 'L3asn.json')
            lev3asnname = 'Level3_' + target + '_NOBG_IFUALIGN_XY0p1'
            writel3asn(calfiles, None, asnfile, lev3asnname)
            # I guess this is deprecated since I put it all in the loop below to be able to do multiple resolutions
            # oh well
                    
    # make multiple cubes at different resolutions:
    if runspec3 == True:
        reslist = cube_resolutions
        print(f'Starting Spec3 Pipeline for {target}')
        for res in reslist:
            print(f'Starting res: {res}')
            spec3 = Spec3Pipeline()
            spec3_params = deepcopy(spec3_param_nested_dict)
            spec3_params['output_dir'] = output_path + folders_L3[0]
            spec3_params['cube_build']['scalexy'] = res
            if target == 'SUNBURST':
                calpath = os.path.join(output_path, folders_L2b[1]) # for convenience
                res_str = str(res).split('.')[-1]
                calfiles_p1 = glob.glob(calpath + '*_02101_*cal.fits') + glob.glob(calpath + '*_10101_*cal.fits')
                calfiles_p2 = glob.glob(calpath + '*_04101_*cal.fits') + glob.glob(calpath + '*_12101_*cal.fits')
                calfiles_p3 = glob.glob(calpath + '*_06101_*cal.fits') + glob.glob(calpath + '*_14101_*cal.fits')
                pos1asnfile = os.path.join(output_path, folders_L3[0], 'P1asn.json')
                lev3asnnameP1 = 'Level3_'+target+'_P1_NOBG_SKYALIGN_XY0p'+res_str
                writel3asn(calfiles_p1, None, pos1asnfile, lev3asnnameP1)
                pos23asnfile = os.path.join(output_path, folders_L3[0], 'P2P3asn.json')
                lev3asnnameP23 = 'Level3_'+target+'_P2P3_NOBG_SKYALIGN_XY0p'+res_str
                writel3asn(calfiles_p2+calfiles_p3, None, pos23asnfile, lev3asnnameP23)
                asnlist = [pos1asnfile,pos23asnfile]
                offsetfileP1 = os.path.join(output_path, fold, 'offsetsP1.asdf')
                make_offset_file(target, calfiles_p1, offsetfileP1)
                offsetfileP23 = os.path.join(output_path, fold, 'offsetsP2P3.asdf')
                make_offset_file(target, calfiles_p2+calfiles_p3, offsetfileP23)
            else:
                fold = folders_L3[0]
                asnfile = os.path.join(output_path, fold, 'L3asn.json')
                res_str = str(res).split('.')[-1]
                lev3asnname = 'Level3_' + target + '_NOBG_NIRCAMALIGN_XY0p'+res_str 
                writel3asn(calfiles, None, asnfile, lev3asnname)
                offsetfile = os.path.join(output_path, fold, 'offsets.asdf')
                make_offset_file(target, calfiles, offsetfile)
                ###############################################
            # Each line is: target, x pixels, y pixels, RA center, Dec center.
            cube_sizes = [
                ('COSMICEYE', 59, 65, 323.8028721, -1.0287456),
                ('SGAS1110',  59, 79, 167.5832686, 64.9974530),
                ('SGAS1429',  55, 51, 217.4787892, 12.0439266),
                ('SGAS1527',  53, 53, 231.9388828, 6.8719792),
                ('SGAS1050', 109, 79, 162.6633975, 0.2910689),
                ('SGAS2111',  79, 75, 317.8276240, -1.2412612),
            ]

            for target_name, nspax_x, nspax_y, ra_center, dec_center in cube_sizes:
                if target == target_name:
                    spec3_params['cube_build'].update({
                        'nspax_x': nspax_x,
                        'nspax_y': nspax_y,
                        'ra_center': ra_center,
                        'dec_center': dec_center,
                    })
                    break
            if target != 'SUNBURST':
                spec3_params['cube_build']['offset_file'] = offsetfile
                            ###############################################
            apply_pipeline_parameters(spec3, spec3_params)
            if target == 'SUNBURST':
                for i,asn in enumerate(asnlist):
                    ##### COMMENTED ONLY FOR ONE RUN - ADD BACK IN FOR NEXT PRODUCTION VERSION
                    #if i == 0:
                    #   # P1
                    #   spec3.cube_build.nspax_x = 67
                    #   spec3.cube_build.nspax_y = 61
                    #   spec3.cube_build.ra_center  = 237.5198779
                    #   spec3.cube_build.dec_center = -78.1830012
                    #   spec3.cube_build.offset_file = offsetfileP1
                    #if i == 1:
                    #   # P2+P3
                    #   spec3.cube_build.nspax_x = 65
                    #   spec3.cube_build.nspax_y = 75
                    #   spec3.cube_build.ra_center  = 237.5008791
                    #   spec3.cube_build.dec_center = -78.1864364
                    #   spec3.cube_build.offset_file = offsetfileP23
                    run_jwst_spec3(spec3, asn)
                    # run bg subtraction:
                    if run_bgsub:
                        do_bgsub_step(target, output_path, folders_L3, lev3asnnameP1)
                        do_bgsub_step(target, output_path, folders_L3, lev3asnnameP23)
            else:
                run_jwst_spec3(spec3, asnfile)
                # run bg subtraction:
                if run_bgsub:
                    do_bgsub_step(target, output_path, folders_L3, lev3asnname)
                print(f'Done res: {res}')

        if target == 'SGAS1110': # handle wopr:
            calfiles2 = glob.glob(output_path + folders_L2b[1] + 'jw03843005*cal.fits') # wopr only = obs 005
            if not calfiles2:
                print(
                    'Skipping WOPR-only Spec3 processing: no files matched '
                    f'{os.path.join(output_path, folders_L2b[1], "jw03843005*cal.fits")}'
                )
            for res in reslist if calfiles2 else ():
                print(f'Starting wopr-only res: {res}')
                fold = folders_L3[0]
                asnfile = os.path.join(output_path, fold, 'L3asn.json')
                res_str = str(res).split('.')[-1]
                lev3asnname = 'Level3_' + target + '_NOBG_NIRCAMALIGN_WOPRONLY_XY0p'+res_str
                writel3asn(calfiles2, None, asnfile, lev3asnname)
                offsetfile2 = os.path.join(output_path, fold, 'offsets_wopr.asdf')
                make_offset_file(target, calfiles2, offsetfile2)
                spec3 = Spec3Pipeline()
                spec3_params = deepcopy(spec3_param_nested_dict)
                spec3_params['output_dir'] = output_path + fold
                spec3_params['cube_build'].update({
                    'scalexy': res,
                    'nspax_x': 59,
                    'nspax_y': 79,
                    'ra_center': 167.5832686,
                    'dec_center': 64.9974530,
                    'offset_file': offsetfile2,
                })
                apply_pipeline_parameters(spec3, spec3_params)
                run_jwst_spec3(spec3, asnfile)
                # run bg subtraction:
                if run_bgsub:
                    do_bgsub_step(target, output_path, folders_L3, lev3asnname, special=True)
                print(f'Done wopr-only res: {res}')

        print(f'Done Spec3 Pipeline for {target}')
                    # and now, run spec3 cube building with non-linear prism wavelength solutions:
    if runspec3_PRISM:
        # get calfiles for prism only:
        calfiles_prism = []
        for file in calfiles:
            with fits.open(file) as hdu:
                if hdu[0].header['GRATING'] == 'PRISM':
                    calfiles_prism.append(file)
                    # print(calfiles_prism)
                    asnfile = os.path.join(output_path, fold, 'L3PRISMasn.json')
                    reslist = [0.1]#, 0.03]
        for res in reslist:
            print(f'Starting non-linear prism, res {res}')
            res_str = str(res).split('.')[-1]
            lev3asnname = 'Level3_' + target + '_NOBG_MULTICUBE_XY0p'+res_str
            writel3asn(calfiles_prism, None, asnfile, lev3asnname)
            spec3 = Spec3Pipeline()
            spec3_params = deepcopy(spec3_param_nested_dict)
            spec3_params['output_dir'] = output_path + fold
            spec3_params['cube_build'].update({
                'coord_system': 'ifualign',
                'scalexy': res,
                'output_type': 'multi',
            })
            apply_pipeline_parameters(spec3, spec3_params)
            run_jwst_spec3(spec3, asnfile)
            print(f'Done res {res}')





###############################################
## HELPER FUNCTIONS FOR RUNNING THE PIPELINE ##
###############################################


# BELOW COPIED FROM DAVID LAW'S MIRI MRS NOTEBOOK: 
# https://github.com/STScI-MIRI/MRS-ExampleNB/blob/main/Flight_Notebook1/MRS_FlightNB1.ipynb
# 
# Define a useful function to write out a Lvl3 association file from an input list
# Note that any background exposures have to be of type x1d.
def writel3asn(scifiles, bgfiles, asnfile, prodname):
    # Define the basic association of science files
    asn = afl.asn_from_list(scifiles, rule=DMS_Level3_Base, product_name=prodname)

    # Add background files to the association
    if bgfiles:
        nbg=len(bgfiles)
        for ii in range(0,nbg):
            asn['products'][0]['members'].append({'expname': bgfiles[ii], 'exptype': 'background'})
        
    # Write the association to a json file
    _, serialized = asn.dump()
    with open(asnfile, 'w') as outfile:
        outfile.write(serialized)

def make_offset_file(target, calfiles, offsetfile):
    ra_offsets  = []
    dec_offsets = []
    filelist    = []
    if target == 'COSMICEYE':
        prism_dra  = 0.04427999992913101
        prism_ddec = 0.13320000000023313
        g235m_dra  = -0.016199999959098932
        g235m_ddec = -0.16164000000005174
    if target == 'SGAS1429':
        prism_dra  = 0.0748
        prism_ddec = 0.1256
        g235m_dra  = 0.0748
        g235m_ddec = 0.1256 
    if target == 'SGAS1110':
        prism_dra  = 0.4 - 0.2
        prism_ddec = 0.01368
        g235m_dra  = 0.4 - 0.2
        g235m_ddec = 0.01368
        g140h_dra  = 0.11 - 0.05
        g140h_ddec = -0.14
    if target == 'SGAS1527':
        prism_dra  = 0.154
        prism_ddec = 0.065
        g235m_dra  = 0.104 
        g235m_ddec = 0.103
    if target == 'SGAS1050':
        prism_dra  = 0.09
        prism_ddec = 0.1184
        g235m_dra  = 0.084
        g235m_ddec = 0.1116
    if target == 'SGAS2111':
        prism_dra  = -0.039
        prism_ddec = -0.134
        g235m_dra  = -0.039
        g235m_ddec = -0.134
    if target == 'SUNBURST':
        g235m_dra  = -1.25#6.37
        g235m_ddec = 0.48
        g140h_dra  = -1.25#6.37
        g140h_ddec = 0.48
        #print(calfiles)
    for file in calfiles:
        with fits.open(file) as hdu:
            head = hdu[0].header.copy()
        grating = head['GRATING']
        if grating == 'PRISM':
            ra_offsets.append(prism_dra)
            dec_offsets.append(prism_ddec)
        if grating == 'G235M':
            ra_offsets.append(g235m_dra)
            dec_offsets.append(g235m_ddec)
        if grating == 'G140H':
            ra_offsets.append(g140h_dra)
            dec_offsets.append(g140h_ddec)
        if grating == 'G235H':
            ra_offsets.append(g235m_dra) # use g235m also for g235h - they're close enough by eye
            dec_offsets.append(g235m_ddec)
        if grating in ('PRISM', 'G235M', 'G140H', 'G235H'):
            filelist.append(os.path.basename(file))
    tree = {
        "units": str(u.arcsec),
        "filename":filelist,
        "raoffset": ra_offsets,
        "decoffset": dec_offsets
    }
    with asdf.AsdfFile(tree) as af:
        af.write_to(offsetfile)


def cut_cal(infile, outfile, max_threshold=20, min_threshold=-1):
    hdu=fits.open(infile)
    sci=hdu['SCI'].data
    dq=hdu['DQ'].data

    dnubit=dqflags.interpret_bit_flags('DO_NOT_USE', mnemonic_map=datamodels.dqflags.pixel)
    indx=np.where((dq & dnubit) != 0)
    sci[indx]=np.nan

    indx=np.where((sci > max_threshold) | (sci < min_threshold))
    sci[indx]=np.nan
    dq[indx] = np.bitwise_or(dq[indx], dnubit)

    hdu['SCI'].data=sci
    hdu.writeto(outfile, overwrite=True)



def run_clean_step(file, clean_dir, expand_flat=False):
    DO_NOT_USE = datamodels.dqflags.pixel["DO_NOT_USE"]
    NON_SCIENCE= datamodels.dqflags.pixel["NON_SCIENCE"]
    NO_SAT_CHECK = datamodels.dqflags.pixel["NO_SAT_CHECK"] # 2^21 = 2097152
    UNRELIABLE_FLAT = datamodels.dqflags.pixel["UNRELIABLE_FLAT"] # 2^25 = 33554432
    OTHER_BAD_PIXEL = datamodels.dqflags.pixel['OTHER_BAD_PIXEL'] # 2^30 = 1073741824
    TELEGRAPH_PIXEL = datamodels.dqflags.pixel['TELEGRAPH'] # 2^15 = 32768
    MSA_FAILED_OPEN = datamodels.dqflags.pixel['MSA_FAILED_OPEN'] # 2^29 = 536870912
    FLUX_ESTIMATED = datamodels.dqflags.pixel['FLUX_ESTIMATED'] # 2^28 = 268435456
    UNRELIABLE_DARK = datamodels.dqflags.pixel['UNRELIABLE_DARK'] # 2^23 = 8388608

    filename = os.path.basename(file)
    #filename = filename.replace('cal2','cal3') # don't update the names for this run, to save some disk space

    new_file = clean_dir + filename  # write to a different directory to keep things straight
    test_file = filename.replace('.fits','test.fits') # this is just for testing and looking at where the new bad pixels have 
    # been flagged
    new_test_file = clean_dir + test_file
    print('test file', new_test_file)

    print('new file', new_file)
    input_cal = datamodels.IFUImageModel(file)
    input_test = input_cal.copy()

    non_science = np.bitwise_and(input_cal.dq, NON_SCIENCE).astype(bool)
    bad = np.bitwise_and(input_cal.dq, DO_NOT_USE).astype(bool)
    num_science = np.where(~non_science & ~bad)
    num_science = len(num_science[0])
    print('number of good science pixels at the start', num_science)
    loc_non_science = np.where(np.bitwise_and(input_cal.dq, NON_SCIENCE).astype(bool))

    data1 = input_cal.data
    dq1 = input_cal.dq
    dq1_donotuse = np.where(np.bitwise_and(dq1, DO_NOT_USE).astype(bool))
    dq_test =  dq1.copy() * 0 # initialize to 0 
    dq_test[dq1_donotuse] = 1 # set up bad pixels
    dq_test[loc_non_science] = 1 

    uflat = np.bitwise_and(dq1, UNRELIABLE_FLAT).astype(bool)
    bad =  np.bitwise_and(dq1, DO_NOT_USE).astype(bool)
    non_science =  np.bitwise_and(dq1, NON_SCIENCE).astype(bool)
    test1 = np.where( uflat & ~bad & ~non_science)
    num = len(test1[0])

    # flag new unreliable flat 
    dq_test[test1] = 3  
    ##

    if num > 0:
        print('Number of pixels with UNREIABLE_FLAT but no DO_NOT_USE', num)
        print('% valid pixels',(num/num_science)* 100)
        input_cal.dq[test1] = np.bitwise_or(input_cal.dq[test1], datamodels.dqflags.pixel['DO_NOT_USE'])
        input_cal.data[test1] = np.nan


    # NO_SAT_CHECK
    data2 = input_cal.data
    dq2 = input_cal.dq

    no_sat = np.bitwise_and(dq2, NO_SAT_CHECK).astype(bool)
    bad =  np.bitwise_and(dq2, DO_NOT_USE).astype(bool)
    non_science =  np.bitwise_and(dq2, NON_SCIENCE).astype(bool) # check if it is a NON-Science pixel
    test2 = np.where( no_sat & ~bad & ~non_science)
    num = len(test2[0])

    # set test DQ
    dq_test[test2] = 5
    ##

    if num > 0:
        print('Number of pixels with NO_SAT_CHECK but no DO_NOT_USE ', num)
        print('% valid pixels',(num/num_science)* 100)
        input_cal.dq[test2] = np.bitwise_or(input_cal.dq[test2], datamodels.dqflags.pixel['DO_NOT_USE']) 
        input_cal.data[test2] = np.nan


    # OTHER_BAD_PIXEL 
    data3 = input_cal.data
    dq3 = input_cal.dq

    other_bad = np.bitwise_and(dq3, OTHER_BAD_PIXEL).astype(bool)
    bad =  np.bitwise_and(dq3, DO_NOT_USE).astype(bool)
    non_science =  np.bitwise_and(dq3, NON_SCIENCE).astype(bool)
    test3 = np.where( other_bad & ~bad & ~non_science)
    num = len(test3[0])

    # set test DQ
    dq_test[test3] = 7
    ##


    if num > 0:
        print('Number of pixels with OTHER_BAD_PIXEL but not DO_NOT_USE', num)
        print('% valid pixels',(num/num_science)* 100)
        input_cal.dq[test3] = np.bitwise_or(input_cal.dq[test3], datamodels.dqflags.pixel['DO_NOT_USE'])
        input_cal.data[test3] = np.nan

    # TELEGRAPH Pixel
    data4 = input_cal.data
    dq4 = input_cal.dq

    # set test DQ
    tp = np.bitwise_and(dq4, TELEGRAPH_PIXEL).astype(bool)
    bad =  np.bitwise_and(dq4, DO_NOT_USE).astype(bool)
    non_science =  np.bitwise_and(dq4, NON_SCIENCE).astype(bool)
    test4 = np.where( tp & ~bad & ~non_science)
    num = len(test4[0])

    #
    dq_test[test4] = 9
    ##


    if num > 0:  
        print('Number of pixels with Random Telegraph  but no DO_NOT_USE', num)
        print('% valid pixels',(num/num_science)* 100)
        input_cal.dq[test4] = np.bitwise_or(input_cal.dq[test4], datamodels.dqflags.pixel['DO_NOT_USE'])
        input_cal.data[test4] = np.nan


    # MSA_FAILED_OPEN Pixel
    data5 = input_cal.data
    dq5 = input_cal.dq

    # set test DQ
    msa = np.bitwise_and(dq5, MSA_FAILED_OPEN).astype(bool)
    bad =  np.bitwise_and(dq5, DO_NOT_USE).astype(bool)
    non_science =  np.bitwise_and(dq5, NON_SCIENCE).astype(bool)
    test5 = np.where( msa & ~bad & ~non_science)
    num = len(test5[0])

    #
    dq_test[test5] = 11
    ##


    if num > 0:  
        print('Number of pixels with Failed Open Shutter but no DO_NOT_USE', num)
        print('% valid pixels',(num/num_science)* 100)
        input_cal.dq[test5] = np.bitwise_or(input_cal.dq[test5], datamodels.dqflags.pixel['DO_NOT_USE'])
        input_cal.data[test5] = np.nan


    # FLUX_ESTIMATED Pixel
    data6 = input_cal.data
    dq6 = input_cal.dq

    # set test DQ
    fe = np.bitwise_and(dq6, FLUX_ESTIMATED).astype(bool)
    bad =  np.bitwise_and(dq6, DO_NOT_USE).astype(bool)
    non_science =  np.bitwise_and(dq6, NON_SCIENCE).astype(bool)
    test6 = np.where( fe & ~bad & ~non_science)
    num = len(test6[0])

    #
    dq_test[test6] = 13
    ##


    if num > 0:  
        print('Number of pixels with Flux Estimated  but no DO_NOT_USE', num)
        print('% valid pixels',(num/num_science)* 100)
        input_cal.dq[test6] = np.bitwise_or(input_cal.dq[test6], datamodels.dqflags.pixel['DO_NOT_USE'])
        input_cal.data[test6] = np.nan


    # UNRELIABLE_DARK Pixel
    data7 = input_cal.data
    dq7 = input_cal.dq

    # set test DQ
    ud = np.bitwise_and(dq7, UNRELIABLE_DARK).astype(bool)
    bad =  np.bitwise_and(dq7, DO_NOT_USE).astype(bool)
    non_science =  np.bitwise_and(dq7, NON_SCIENCE).astype(bool)
    test7 = np.where( ud & ~bad & ~non_science)
    num = len(test7[0])

    #
    dq_test[test7] = 14
    ##


    if num > 0:  
        print('Number of pixels with Unreliable Dark but no DO_NOT_USE', num)
        print('% valid pixels',(num/num_science)* 100)
        input_cal.dq[test7] = np.bitwise_or(input_cal.dq[test7], datamodels.dqflags.pixel['DO_NOT_USE'])
        input_cal.data[test7] = np.nan


    # This one may not be need. You should check
    # now expand the unrelaible_flat flag to be 1 pixel more along the edge of the slices
    #expand_flat = True
    if expand_flat:
        dq_expand = input_cal.dq
        uflat = np.where(np.bitwise_and(dq_expand, UNRELIABLE_FLAT).astype(bool))
        nflat = len(uflat[0])
        # loop over the pixels with UNRELIABLE FLAT set. Check y+1, y-1 (not sure which edge of the slice we
        # on) and then these pixels to UNRELIABLE_FLAT
        print('Looping over ', nflat)

        for i in range (nflat):
            ix = uflat[1][i]
            iy = uflat[0][i]
            input_cal.dq[iy+1, ix] = np.bitwise_or(input_cal.dq[iy+1,ix], datamodels.dqflags.pixel['DO_NOT_USE'])
            input_cal.dq[iy+1, ix] = np.bitwise_or(input_cal.dq[iy+1,ix], datamodels.dqflags.pixel['UNRELIABLE_FLAT'])
                    
            input_cal.dq[iy-1, ix] = np.bitwise_or(input_cal.dq[iy-1,ix], datamodels.dqflags.pixel['DO_NOT_USE'])
            input_cal.dq[iy-1, ix] = np.bitwise_or(input_cal.dq[iy-1,ix], datamodels.dqflags.pixel['UNRELIABLE_FLAT'])
                    
            input_cal.dq[iy-2, ix] = np.bitwise_or(input_cal.dq[iy-2,ix], datamodels.dqflags.pixel['DO_NOT_USE'])
            input_cal.dq[iy-2, ix] = np.bitwise_or(input_cal.dq[iy-2,ix], datamodels.dqflags.pixel['UNRELIABLE_FLAT'])
                    
            input_cal.dq[iy+2, ix] = np.bitwise_or(input_cal.dq[iy+2,ix], datamodels.dqflags.pixel['DO_NOT_USE'])
            input_cal.dq[iy+2, ix] = np.bitwise_or(input_cal.dq[iy+2,ix], datamodels.dqflags.pixel['UNRELIABLE_FLAT'])
                    

    input_cal.save(new_file)
    input_test.dq = dq_test
    #input_test.save(new_test_file) # you can view the DQ plane in DS9 and easily see additional pixels that are flagged as bad
    print(test_file)




def exp_bg_sub(cubefile, bgfile, outfile=None):
    # first, load the expected background data:
    exp = np.loadtxt(bgfile)
    expwl = np.array([line[0] for line in exp])
    exptot = np.array([line[1] for line in exp])
    # and make an interp1d object sdo we can get the background at any given point:
    bginterp = interp1d(expwl, exptot)
    # now load the data cube:
    hdu = fits.open(cubefile)
    data = hdu[1].data
    head1 = hdu[1].header
    cubewl = np.arange(head1['CRVAL3'],
                 head1['CRVAL3']+(head1['CDELT3']*len(data)),
                 head1['CDELT3'])
    if len(cubewl) > len(data):
        cubewl = cubewl[:-1]
    # next, evaluate background on data wavelength points:
    bgcalculated = bginterp(cubewl)
    bgcube = np.tile(bgcalculated, (data.shape[2],data.shape[1],1)).T # cubeify!
    # and subtract!
    #print(len(data), len(cubewl))
    bgsub_cube = data - bgcube
    # save or return result:
    if outfile:
        hdu[1].data = bgsub_cube
        hdu[1].header['comment'] = 'Expected Background Subtracted'
        hdu.writeto(outfile, overwrite=True)
        hdu.close()
    else:
        hdu.close()
        return bgsub_cube


def do_bgsub_step(target, output_path, folders_L3, lev3asnname, special=False):
    cube_suffixes = {
        'SGAS1110': ('_g140h-f100lp_s3d.fits', '_g235h-f170lp_s3d.fits',
                     '_g235m-f170lp_s3d.fits'),
        'SGAS2111': ('_g235m-f170lp_s3d.fits',),
        'SGAS1050': ('_g235m-f170lp_s3d.fits',),
        'SGAS1429': ('_g235m-f170lp_s3d.fits',),
        'SGAS1527': ('_g235m-f170lp_s3d.fits',),
        'COSMIC.EYE': ('_g235m-f170lp_s3d.fits',),
        'COSMICEYE': ('_g235m-f170lp_s3d.fits',),
        'SGAS1723': ('_g140h-f100lp_s3d.fits', '_g395h-f290lp_s3d.fits'),
        'SGAS1226': ('_g235h-f170lp_s3d.fits',),
        'SPT0418-47': ('_g395m-f290lp_s3d.fits',),
        'SPT2147-50': ('_g395m-f290lp_s3d.fits',),
        'SPARKLER': ('_g140m-f070lp_s3d.fits',),
        'SUNBURST': ('_g140h-f100lp_s3d.fits', '_g235h-f170lp_s3d.fits'),
    }

    if target not in cube_suffixes:
        return

    expbg_file = os.path.join(dir_bkgfiles, target + '_background.txt')

    suffixes = cube_suffixes[target]
    if target == 'SGAS1110' and special:
        suffixes = ('_g235h-f170lp_s3d.fits',)

    for suffix in suffixes:
        cubefile = os.path.join(output_path, folders_L3[0], lev3asnname + suffix)
        outfile = cubefile.replace('NOBG', 'BGSUB')
        exp_bg_sub(cubefile, expbg_file, outfile)


