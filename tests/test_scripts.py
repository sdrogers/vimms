import os
import pytest
from loguru import logger
from vimms.Chemicals import ChemicalMixtureCreator, MultipleMixtureCreator
from vimms.ChemicalSamplers import UniformRTAndIntensitySampler, UniformMZFormulaSampler
from tests.conftest import OUT_DIR
from vimms.Utils import write_msp
from vimms.Controller import TopNController
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Common import POSITIVE, set_log_level_warning, set_log_level_debug
from vimms.scripts.check_ms2_matches import main as ms2_main

class TestMS2Matching:
    def test_ms2_matching(self):
        rti = UniformRTAndIntensitySampler(min_rt=10, max_rt=20)
        fs = UniformMZFormulaSampler()
        adduct_prior_dict = {POSITIVE: {'M+H': 1}}

        cs = ChemicalMixtureCreator(fs, rt_and_intensity_sampler=rti, adduct_prior_dict=adduct_prior_dict)
        d = cs.sample(300,2)

        group_list = ['control', 'control', 'case', 'case']
        group_dict = {}
        group_dict['control'] = {'missing_probability': 0.0, 'changing_probability': 0.0}
        group_dict['case'] = {'missing_probability': 0.0, 'changing_probability': 1.0}

        mm = MultipleMixtureCreator(d, group_list, group_dict)

        cl = mm.generate_chemical_lists()


        N = 10
        isolation_width = 0.7
        mz_tol = 0.001
        rt_tol = 30
        min_ms1_intensity = 0

        set_log_level_warning()


        output_folder = os.path.join(OUT_DIR, 'ms2_matching')
        write_msp(d, 'mmm.msp', out_dir=output_folder)

        initial_exclusion_list = None
        for i, chem_list in enumerate(cl):
            controller = TopNController(POSITIVE, N, isolation_width, \
                                        mz_tol, rt_tol, min_ms1_intensity, \
                                        initial_exclusion_list=initial_exclusion_list)
            ms = IndependentMassSpectrometer(POSITIVE, chem_list, None)
            env = Environment(ms, controller, 10, 30, progress_bar=True)
            env.run()
            env.write_mzML(output_folder,'{}.mzML'.format(i))

            if initial_exclusion_list is None:
                initial_exclusion_list = []
            initial_exclusion_list = initial_exclusion_list + controller.all_exclusion_items
            logger.warning(len(initial_exclusion_list))


        set_log_level_debug()
        msp_file = os.path.join(output_folder,'mmm.msp')
        # check with just the first file
        a, b = ms2_main(os.path.join(output_folder, '0.mzML'), msp_file, 1, 0.7)
        # check with all
        c, d = ms2_main(output_folder, os.path.join(output_folder,'mmm.msp'), 1, 0.7)
    
        assert b==d
        assert c > a


