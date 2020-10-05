from tests.conftest import OUT_DIR, check_mzML
from vimms.Agent import TopNDEWAgent
from vimms.ChemicalSamplers import UniformMZFormulaSampler, UniformRTAndIntensitySampler, GaussianChromatogramSampler, \
    FixedMS2Sampler
from vimms.Chemicals import ChemicalMixtureCreator
from vimms.Common import POSITIVE, set_log_level_warning, set_log_level_debug
from vimms.Controller import AgentBasedController
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Noise import UniformSpikeNoise


class TestAgentBasedController:
    def test_TopNDEW_agent(self):
        set_log_level_debug()
        fs = UniformMZFormulaSampler()
        ri = UniformRTAndIntensitySampler(min_rt=0, max_rt=80)
        cr = GaussianChromatogramSampler(sigma=1)
        ms = FixedMS2Sampler()
        cs = ChemicalMixtureCreator(fs, rt_and_intensity_sampler=ri, chromatogram_sampler=cr, ms2_sampler=ms)
        d = cs.sample(500, 2)
        ionisation_mode = POSITIVE

        # Example shows how the same Agent object can be used in consecutive controllers

        agent = TopNDEWAgent(10, 1000, 10, 1500)
        controller = AgentBasedController(ionisation_mode, agent)
        spike_noise = UniformSpikeNoise(0.1, 1000)
        mass_spec = IndependentMassSpectrometer(ionisation_mode, d, None, spike_noise=spike_noise)
        env = Environment(mass_spec, controller, 0, 100, progress_bar=True)
        set_log_level_warning()
        env.run()

        for level in controller.scans:
            for scan in controller.scans[level]:
                assert len(scan.mzs) > 0, scan.scan_id

        check_mzML(env, OUT_DIR, 'shell.mzML')

        controller = AgentBasedController(ionisation_mode, agent)
        mass_spec = IndependentMassSpectrometer(ionisation_mode, d, None, spike_noise=spike_noise)
        env = Environment(mass_spec, controller, 0, 100, progress_bar=True)
        set_log_level_warning()
        env.run()

        check_mzML(env, OUT_DIR, 'shell2.mzML')

        controller = AgentBasedController(ionisation_mode, agent)
        mass_spec = IndependentMassSpectrometer(ionisation_mode, d, None, spike_noise=spike_noise)
        env = Environment(mass_spec, controller, 0, 100, progress_bar=True)
        set_log_level_warning()
        env.run()

        check_mzML(env, OUT_DIR, 'shell3.mzML')
