import math

import numpy as np
import scipy
from events import Events
from loguru import logger

from vimms.Common import adduct_transformation, DEFAULT_SCAN_TIME_DICT, INITIAL_SCAN_ID, ScanParameters
from vimms.Noise import NoPeakNoise


class Peak(object):
    """
    A class to represent an empirical or sampled scan-level peak object
    """

    def __init__(self, mz, rt, intensity, ms_level):
        """
        Creates a peak object
        :param mz: mass-to-charge value
        :param rt: retention time value
        :param intensity: intensity value
        :param ms_level: MS level
        """
        self.mz = mz
        self.rt = rt
        self.intensity = intensity
        self.ms_level = ms_level

    def __repr__(self):
        return 'Peak mz=%.4f rt=%.2f intensity=%.2f ms_level=%d' % (self.mz, self.rt, self.intensity, self.ms_level)

    def __eq__(self, other):
        if not isinstance(other, Peak):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return math.isclose(self.mz, other.mz) and \
               math.isclose(self.rt, other.rt) and \
               math.isclose(self.intensity, other.intensity) and \
               self.ms_level == other.ms_level


class Scan(object):
    """
    A class to store scan information
    """

    def __init__(self, scan_id, mzs, intensities, ms_level, rt,
                 scan_duration=None, scan_params=None, parent=None):
        """
        Creates a scan
        :param scan_id: current scan id
        :param mzs: an array of mz values
        :param intensities: an array of intensity values
        :param ms_level: the ms level of this scan
        :param rt: the retention time of this scan
        :param scan_duration: how long this scan takes, if known.
        :param scan_params: the parameters used to generate this scan, if known
        :param parent: parent precursor peak, if known
        """
        assert len(mzs) == len(intensities)
        self.scan_id = scan_id

        # ensure that mzs and intensites are sorted by their mz values
        p = mzs.argsort()
        self.mzs = mzs[p]
        self.intensities = intensities[p]

        self.ms_level = ms_level
        self.rt = rt
        self.num_peaks = len(mzs)

        self.scan_duration = scan_duration
        self.scan_params = scan_params
        self.parent = parent

    def __repr__(self):
        return 'Scan %d num_peaks=%d rt=%.2f ms_level=%d' % (self.scan_id, self.num_peaks, self.rt, self.ms_level)


class ScanEvent(object):
    """
    A class to store fragmentation events. Mostly used for benchmarking purpose.
    """

    def __init__(self, chem, query_rt, ms_level, peaks, scan_id, parents_intensity=None, parent_adduct=None,
                 parent_isotope=None, precursor_mz=None, isolation_window=None, scan_params=None):
        """
        Creates a fragmentation event
        :param chem: the chemical that were fragmented
        :param query_rt: the time when fragmentation occurs
        :param ms_level: MS level of fragmentation
        :param peaks: the set of peaks produced during the fragmentation event
        :param scan_id: the scan id linked to this fragmentation event
        :param parents_intensity: the intensity of the chemical that was fragmented at the time it was fragmented
        :param parent_adduct: the adduct that was fragmented of the chemical
        :param parent_isotope: the isotope that was fragmented of the chemical
        :param precursor_mz: the precursor mz of the scan
        :param isolation_window: the isolation window of the scan
        :param scan_params: the scan parameter settings that were used
        """
        self.chem = chem
        self.query_rt = query_rt
        self.ms_level = ms_level
        self.peaks = peaks
        self.scan_id = scan_id
        self.parents_intensity = parents_intensity   # only ms2
        self.parent_adduct = parent_adduct   # only ms2
        self.parent_isotope = parent_isotope   # only ms2
        self.precursor_mz = precursor_mz
        self.isolation_window = isolation_window
        self.scan_params = scan_params

    def __repr__(self):
        return 'MS%d ScanEvent for %s at %f' % (self.ms_level, self.chem, self.query_rt)


class IndependentMassSpectrometer(object):
    """
    A class that represents (synchronous) mass spectrometry process.
    Independent here refers to how the intensity of each peak in a scan is independent of each other
    i.e. there's no ion supression effect.
    """
    MS_SCAN_ARRIVED = 'MsScanArrived'
    ACQUISITION_STREAM_OPENING = 'AcquisitionStreamOpening'
    ACQUISITION_STREAM_CLOSED = 'AcquisitionStreamClosing'
    STATE_CHANGED = 'StateChanged'

    def __init__(self, ionisation_mode, chemicals, peak_sampler, mz_noise=None, intensity_noise=None, spike_noise=None,
                 isolation_transition_window='rectangular', isolation_transition_window_params=None,
                 scan_duration_dict=DEFAULT_SCAN_TIME_DICT):
        """
        Creates a mass spec object.
        :param ionisation_mode: POSITIVE or NEGATIVE
        :param chemicals: a list of Chemical objects in the dataset
        :param peak_sampler: an instance of DataGenerator.PeakSampler object
        :param add_noise: a flag to indicate whether to add noise
        :param use_exclusion_list: a flag to indicate whether to perform dynamic exclusion
        """

        # current scan index and internal time
        self.idx = INITIAL_SCAN_ID  # same as the real mass spec
        self.time = 0

        # current task queue
        self.processing_queue = []
        self.environment = None

        self.events = Events((self.MS_SCAN_ARRIVED, self.ACQUISITION_STREAM_OPENING, self.ACQUISITION_STREAM_CLOSED,
                              self.STATE_CHANGED,))
        self.event_dict = {
            self.MS_SCAN_ARRIVED: self.events.MsScanArrived,
            self.ACQUISITION_STREAM_OPENING: self.events.AcquisitionStreamOpening,
            self.ACQUISITION_STREAM_CLOSED: self.events.AcquisitionStreamClosing,
            self.STATE_CHANGED: self.events.StateChanged
        }

        # the list of all chemicals in the dataset
        self.chemicals = chemicals
        self.ionisation_mode = ionisation_mode  

        # stores the chromatograms start and end rt for quick retrieval
        chem_rts = np.array([chem.rt for chem in self.chemicals])
        self.chrom_min_rts = np.array([chem.chromatogram.min_rt for chem in self.chemicals]) + chem_rts
        self.chrom_max_rts = np.array([chem.chromatogram.max_rt for chem in self.chemicals]) + chem_rts

        # here's where we store all the stuff to sample from
        self.peak_sampler = peak_sampler

        # required to sample for different scan durations based on (N, DEW) in the hybrid controller
        self.current_N = 0
        self.current_DEW = 0

        # whether to add noise to the generated peaks, the default is no noise
        self.mz_noise = mz_noise
        self.intensity_noise = intensity_noise
        if self.mz_noise is None:
            self.mz_noise = NoPeakNoise()
        if self.intensity_noise is None:
            self.intensity_noise = NoPeakNoise()
        self.spike_noise = spike_noise

        self.fragmentation_events = []  # which chemicals produce which peaks

        self.isolation_transition_window = isolation_transition_window
        self.isolation_transition_window_params = isolation_transition_window_params

        self.scan_duration_dict = scan_duration_dict

    ####################################################################################################################
    # Public methods
    ####################################################################################################################

    def set_environment(self, env):
        self.environment = env

    def step(self, params=None):
        """
        Performs one step of a mass spectrometry process
        :return:
        """
        # if no params passed in, then try to get one from the queue
        if params is None:
            try:
                # get a scan params from the queue
                params = self.processing_queue.pop(0)
            except IndexError: # nothing in the queue
                params = None

        # if finally we have params, then make a scan
        if params is not None:
            scan = self._get_scan(self.time, params)
            current_level = scan.ms_level

            # notify the controller that a new scan has been generated
            # at this point, the MS_SCAN_ARRIVED event handler in the controller is called
            # and the processing queue will be updated with new sets of scan parameters to do
            self.fire_event(self.MS_SCAN_ARRIVED, scan)

            # sample scan duration and increase internal time
            current_N = self.current_N
            current_DEW = self.current_DEW
            try:
                next_scan_param = self.get_processing_queue()[0]
            except IndexError:
                next_scan_param = None

            current_scan_duration = self._increase_time(current_level, current_N, current_DEW,
                                                        next_scan_param)
            scan.scan_duration = current_scan_duration

            # stores the updated value of N and DEW
            self._store_next_N_DEW(next_scan_param)
            return scan

    def get_processing_queue(self):
        """
        Returns the current processing queue
        :return:
        """
        return self.processing_queue

    def add_to_processing_queue(self, param):
        """
        Adds a new scan parameters to the processing queue of scan parameters. Usually done by the controllers.
        :param param: the scan parameters to add
        :return: None
        """
        self.processing_queue.append(param)

    def fire_event(self, event_name, arg=None):
        """
        Simulates sending an event
        :param event_name: the event name
        :param arg: the event parameter
        :return: None
        """
        if event_name not in self.event_dict:
            raise ValueError('Unknown event name')

        # pretend to fire the event
        # actually here we just runs the event handler method directly
        e = self.event_dict[event_name]
        if arg is not None:
            e(arg)
        else:
            e()

    def register_event(self, event_name, handler):
        """
        Register event handler
        :param event_name: the event name
        :param handler: the event handler
        :return: None
        """
        if event_name not in self.event_dict:
            raise ValueError('Unknown event name')
        e = self.event_dict[event_name]
        e += handler  # register a new event handler for e

    def clear_events(self):
        for key in self.event_dict:  # clear event handlers
            self.clear_event(key)

    def clear_event(self, event_name):
        """
        Clears event handler for a given event name
        :param event_name: the event name
        :return: None
        """
        if event_name not in self.event_dict:
            raise ValueError('Unknown event name')
        e = self.event_dict[event_name]
        e.targets = []

    def close(self):
        logger.debug('Unregistering event handlers')
        self.clear_events()

    ####################################################################################################################
    # Private methods
    ####################################################################################################################

    def _increase_time(self, current_level, current_N, current_DEW, next_scan_param):
        # look into the queue, find out what the next scan ms_level is, and compute the scan duration
        # only applicable for simulated mass spec, since the real mass spec can generate its own scan duration.
        self.idx += 1
        if next_scan_param is None:  # if queue is empty, the next one is an MS1 scan by default
            next_level = 1
        else:
            next_level = next_scan_param.get(ScanParameters.MS_LEVEL)

        # sample current scan duration based on current_DEW, current_N, current_level and next_level
        if self.scan_duration_dict == None:
            current_scan_duration = self._sample_scan_duration(current_DEW, current_N,
                                                               current_level, next_level)
        else:
            val = self.scan_duration_dict[current_level]
            if callable(val):  # is it a function, or a value?
                tt = val()
            else:
                tt = val
            current_scan_duration = tt

        self.time += current_scan_duration
        logger.debug('Time %f Len(queue)=%d' % (self.time, len(self.processing_queue)))
        return current_scan_duration

    def _sample_scan_duration(self, current_DEW, current_N, current_level, next_level):
        # get scan duration based on current and next level
        if current_level == 1 and next_level == 1:
            # special case: for the transition (1, 1), we can try to get the times for the
            # fullscan data (N=0, DEW=0) if it's stored
            try:
                current_scan_duration = self.peak_sampler.scan_durations(current_level, next_level, 1, N=0, DEW=0)
            except KeyError:  ## ooops not found
                current_scan_duration = self.peak_sampler.scan_durations(current_level, next_level, 1,
                                                                         N=current_N, DEW=current_DEW)
        else:  # for (1, 2), (2, 1) and (2, 2)
            current_scan_duration = self.peak_sampler.scan_durations(current_level, next_level, 1,
                                                                     N=current_N, DEW=current_DEW)

        try:
            current_scan_duration = current_scan_duration.flatten()[0]
        except:
            print("Failed to sample, current level =  {}, next level = {}".format(current_level, next_level))
            current_scan_duration = 0.1
        return current_scan_duration

    def _store_next_N_DEW(self, next_scan_param):
        """
        Stores the N and DEW parameter values for the next scan params
        :param next_scan_param: A new set of scan parameters
        :return: None
        """
        if next_scan_param is not None:
            # Only the hybrid controller sends these N and DEW parameters. For other controllers they will be None
            # because N and DEW will never change throughout a run
            next_N = next_scan_param.get(ScanParameters.CURRENT_TOP_N)
            next_DEW = next_scan_param.get(ScanParameters.DYNAMIC_EXCLUSION_RT_TOL)
        else:
            next_N = None
            next_DEW = None

        # keep track of the N and DEW values for the next scan if they have been changed by the Hybrid Controller
        if next_N is not None:
            self.current_N = next_N
        if next_DEW is not None:
            self.current_DEW = next_DEW

    ####################################################################################################################
    # Scan generation methods
    ####################################################################################################################

    def _get_scan(self, scan_time, params):
        """
        Constructs a scan at a particular timepoint
        :param scan_time: the timepoint
        :return: a mass spectrometry scan at that time
        """

        min_measurement_mz = params.get(ScanParameters.FIRST_MASS)
        max_measurement_mz = params.get(ScanParameters.LAST_MASS)

        ms1_source_collision_energy = params.get(ScanParameters.SOURCE_CID_ENERGY)

        scan_mzs = []  # all the mzs values in this scan
        scan_intensities = []  # all the intensity values in this scan
        ms_level = params.get(ScanParameters.MS_LEVEL)
        if ms_level == 1:  # if ms1 then we scan the whole range of m/z
            isolation_windows = params.get(ScanParameters.ISOLATION_WINDOWS)
            if isolation_windows is None:
                isolation_windows = [[(min_measurement_mz, max_measurement_mz)]]
            if not isolation_windows[0][0][0] == min_measurement_mz or not isolation_windows[0][0][
                                                                               1] == max_measurement_mz:
                logger.warning("MS1 scan: MS1 isolation window and measurement mz range mismatch")

        else:  # if ms2 then we check if the isolation window parameter is specified
            # isolation_windows = params.get(ScanParameters.ISOLATION_WINDOWS)
            # if isolation_windows is None:  # if not then we compute from the precursor mz and isolation width
            isolation_windows = params.compute_isolation_windows()

        scan_id = self.idx

        # for all chemicals that come out from the column coupled to the mass spec
        idx = self._get_chem_indices(scan_time)

        # the following is to ensure we generate fragment data when we have a collision energe >0
        use_ms_level = ms_level
        if ms_level == 1 and ms1_source_collision_energy > 0:
            use_ms_level = 2

        for i in idx:
            chemical = self.chemicals[i]
            # mzs is a list of (mz, intensity) for the different adduct/isotopes combinations of a chemical
            mzs = self._get_all_mz_peaks(chemical, scan_time, use_ms_level, isolation_windows)
            peaks = []
            peaks_ms1_intensities = []
            peaks_which_isotopes = []
            peaks_which_adducts = []
            if mzs is not None:
                chem_mzs = []
                chem_intensities = []
                for peak_mz, peak_intensity, peak_ms1_int, peak_isotope, peak_adduct in mzs:
                    if peak_mz >= min_measurement_mz and peak_mz <= max_measurement_mz and peak_intensity > 0:
                        chem_mzs.append(peak_mz)
                        chem_intensities.append(peak_intensity)
                        p = Peak(peak_mz, scan_time, peak_intensity, use_ms_level)
                        peaks.append(p)
                        peaks_ms1_intensities.append(peak_ms1_int)
                        peaks_which_isotopes.append(peak_isotope)
                        peaks_which_adducts.append(peak_adduct)
                scan_mzs.extend(chem_mzs)
                scan_intensities.extend(chem_intensities)
            # for benchmarking purpose
            if len(peaks) > 0:
                frag = ScanEvent(chemical, scan_time, use_ms_level, peaks, scan_id,
                                          parents_intensity=peaks_ms1_intensities,
                                          parent_adduct=peaks_which_adducts,
                                          parent_isotope=peaks_which_isotopes,
                                          precursor_mz=params.get(ScanParameters.PRECURSOR_MZ),
                                          isolation_window=isolation_windows,
                                          scan_params=params)
                self.fragmentation_events.append(frag)

        if self.spike_noise is not None:
            spike_mzs, spike_intensities = self.spike_noise.sample(min_measurement_mz, max_measurement_mz)
            scan_mzs.extend(spike_mzs)
            scan_intensities.extend(spike_intensities)

        scan_mzs = np.array(scan_mzs)
        scan_intensities = np.array(scan_intensities)

        sc = Scan(scan_id, scan_mzs, scan_intensities, ms_level, scan_time, scan_duration=None, scan_params=params)

        # Note: at this point, the scan duration is not set yet because we don't know what the next scan is going to be
        # We will set it later in the get_next_scan() method after we've notified the controller that this scan is produced.
        return sc

    def _get_chem_indices(self, query_rt):
        rtmin_check = self.chrom_min_rts <= query_rt
        rtmax_check = query_rt <= self.chrom_max_rts
        idx = np.nonzero(rtmin_check & rtmax_check)[0]
        return idx

    def _get_all_mz_peaks(self, chemical, query_rt, ms_level, isolation_windows):
        # check if the chemical RT matches the current query RT
        if not self._rt_match(chemical, query_rt):
            return None

        # if yes, then gather all the peaks from all the isotopes and adduct of this chemical
        mz_peaks = []
        for which_isotope in range(len(chemical.isotopes)):
            for which_adduct in range(len(self._get_adducts(chemical))):
                peaks = self._get_mz_peaks(chemical, query_rt, ms_level, isolation_windows, which_isotope, which_adduct)
                mz_peaks.extend(peaks)

        # if no peaks generated, then just return None
        if len(mz_peaks) == 0:
            return None
        # apply noise if any
        noisy_mz_peaks = []
        for i in range(len(mz_peaks)):
            original_mz = mz_peaks[i][0]
            original_intensity = mz_peaks[i][1]
            noisy_mz = self.mz_noise.get(original_mz, ms_level)
            noisy_intensity = self.intensity_noise.get(original_intensity, ms_level)
            noisy_mz_peaks.append((noisy_mz, noisy_intensity, mz_peaks[i][2], mz_peaks[i][3], mz_peaks[i][4]))
        return noisy_mz_peaks

    def _get_mz_peaks(self, chemical, query_rt, ms_level, isolation_windows, which_isotope, which_adduct):
        # EXAMPLE OF USE OF DEFINITION: if we wants to do an ms2 scan on a chemical. we would first have ms_level=2 and the chemicals
        # ms_level =1. So we would go to the "else". We then check the ms1 window matched. It then would loop through
        # the children who have ms_level = 2. So we then go to second elif and return the mz and intensity of each ms2 fragment
        mz_peaks = []
        if ms_level == 1 and chemical.ms_level == 1:  # fragment ms1 peaks
            # returns ms1 peaks if chemical is has ms_level = 1 and scan is an ms1 scan
            if not (which_isotope > 0 and which_adduct > 0):
                # rechecks isolations window if not monoisotopic and "M + H" adduct
                if self._isolation_match(chemical, query_rt, isolation_windows[0], which_isotope, which_adduct):
                    intensity = self._get_intensity(chemical, query_rt, which_isotope, which_adduct)
                    mz = self._get_mz(chemical, query_rt, which_isotope, which_adduct)
                    mz_peaks.extend([(mz, intensity, None, None, None)])
        elif ms_level == chemical.ms_level:
            # returns ms2 fragments if chemical and scan are both ms2, 
            # returns ms3 fragments if chemical and scan are both ms3, etc, etc
            ms1_intensity = self._get_intensity(chemical.parent, query_rt, which_isotope, which_adduct)
            intensity = self._get_intensity(chemical, query_rt, which_isotope, which_adduct)
            mz = self._get_mz(chemical, query_rt, which_isotope, which_adduct)
            if self.isolation_transition_window == 'gaussian':
                parent_mz = self._get_mz(chemical.parent, query_rt, which_isotope, which_adduct)
                scale_factor = scipy.stats.norm(0, self.isolation_transition_window_params[0]).pdf(
                    parent_mz - sum(isolation_windows[ms_level - 2][0]) / 2)
                scale_factor /= scipy.stats.norm(0, self.isolation_transition_window_params[0]).pdf(0)
                intensity *= scale_factor
            return [(mz, intensity, ms1_intensity, which_isotope, which_adduct)]
            # return extra information here for logging
            # TODO: Potential improve how the isotope spectra are generated
        else:
            # check isolation window for ms2+ scans, queries children if isolation windows ok
            if self._isolation_match(chemical, query_rt, isolation_windows[chemical.ms_level - 1], which_isotope,
                                     which_adduct) and chemical.children is not None:
                for i in range(len(chemical.children)):
                    mz_peaks.extend(self._get_mz_peaks(chemical.children[i], query_rt, ms_level, isolation_windows,
                                                       which_isotope, which_adduct))
            else:
                return []
        return mz_peaks

    def _get_adducts(self, chemical):
        if chemical.ms_level == 1:
            if self.ionisation_mode in chemical.adducts:
                return chemical.adducts[self.ionisation_mode]
            else:
                return []
        else:
            return self._get_adducts(chemical.parent)

    def _rt_match(self, chemical, query_rt):
        return chemical.ms_level != 1 or chemical.chromatogram._rt_match(query_rt - chemical.rt)

    def _get_intensity(self, chemical, query_rt, which_isotope, which_adduct):
        if chemical.ms_level == 1:
            intensity = chemical.isotopes[which_isotope][1] * self._get_adducts(chemical)[which_adduct][1] * \
                        chemical.max_intensity
            return intensity * chemical.chromatogram.get_relative_intensity(query_rt - chemical.rt)
        else:
            prop = chemical.parent_mass_prop
            if isinstance(prop, np.ndarray):
                prop = prop[0]
            return self._get_intensity(chemical.parent, query_rt, which_isotope, which_adduct) * \
                   prop * chemical.prop_ms2_mass

    def _get_mz(self, chemical, query_rt, which_isotope, which_adduct):
        if chemical.ms_level == 1:
            return (adduct_transformation(chemical.isotopes[which_isotope][0],
                                          self._get_adducts(chemical)[which_adduct][0]) +
                    chemical.chromatogram.get_relative_mz(query_rt - chemical.rt))
        else:
            ms1_parent = chemical
            while ms1_parent.ms_level != 1:
                ms1_parent = chemical.parent
            isotope_transformation = ms1_parent.isotopes[which_isotope][0] - ms1_parent.isotopes[0][0]
            # TODO: Needs improving
            return (adduct_transformation(chemical.isotopes[0][0],
                                          self._get_adducts(chemical)[which_adduct][0]) + isotope_transformation)

    def _isolation_match(self, chemical, query_rt, isolation_windows, which_isotope, which_adduct):
        # assumes list is formated like:
        # [(min_1,max_1),(min_2,max_2),...]
        for window in isolation_windows:
            if window[0] < self._get_mz(chemical, query_rt, which_isotope, which_adduct) <= window[1]:
                return True
        return False
