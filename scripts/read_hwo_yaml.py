import numpy as np
import astropy.units as u
import os
import yaml
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def load_yaml(fl_path):
    # fl_path assumed to originate from SCI_ENG_DIR environment variable location
    with open(os.getenv("SCI_ENG_DIR") + fl_path, "r") as fl:
        fl_dict = yaml.load(fl, Loader=yaml.SafeLoader)
    return fl_dict


def interp_arr(old_lam, old_vals, new_lam):
    old_lam = old_lam.to(new_lam.unit)
    assert old_lam.unit == new_lam.unit
    interp_func = interp1d(
        old_lam, old_vals, kind="linear", bounds_error=False, fill_value=0
    )
    new_vals = interp_func(new_lam)
    new_vals = np.clip(new_vals, 0, None)  # clip negative values to 0
    return new_vals


reflectivity_path = "/obs_config/reflectivities/"
detectors_path = "/obs_config/Detectors/"


class TELESCOPE:
    def __init__(self, lam):
        self.lam = lam

    def plot(self):
        fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
        axes[0].plot(self.lam, self.total_tele_refl, label="Total telescope refl")
        axes[1].plot(self.lam, self.M1_refl, label="M1_refl")
        axes[1].plot(self.lam, self.M2_refl, label="M2_refl")
        axes[1].set_xlabel("Wavelength [um]")
        axes[0].set_ylabel("Trans/refl")
        axes[1].legend()
        plt.show()

    def load_EAC1(self):

        eac1_fl = "/obs_config/Tel/EAC1.yaml"
        print(f"Loading file: {eac1_fl}")
        eac1_dict = load_yaml(eac1_fl)

        diam_insc = eac1_dict["PM"]["inscribing_diameter"][
            0
        ]  # meters* u.Unit(eac1_dict["PM"]["inscribing_diameter"][1])
        diam_circ = eac1_dict["PM"]["circumscribing_diameter"][
            0
        ]  # meters * u.Unit(eac1_dict["PM"]["segmentation_parameters"]["circumscribing_diameter"][1])

        print("Calculating telescope throughput...")
        # M1 reflectivity
        M1_reflectivity_fl = eac1_dict["PM"]["reflectivity"]
        M1_reflectivity_dict = load_yaml(
            reflectivity_path + M1_reflectivity_fl.split("/")[-1]
        )
        M1_refl = M1_reflectivity_dict["reflectivity"]
        M1_lam = M1_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        M1_refl = interp_arr(M1_lam, M1_refl, self.lam)

        # M2 reflectivity
        M2_reflectivity_fl = eac1_dict["SM"]["reflectivity"]
        M2_reflectivity_dict = load_yaml(
            reflectivity_path + M1_reflectivity_fl.split("/")[-1]
        )
        M2_refl = M2_reflectivity_dict["reflectivity"]
        M2_lam = M2_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        M2_refl = interp_arr(M2_lam, M2_refl, self.lam)

        # M3 reflectivity
        print("Warning: M3 reflectivity not included in YAML")
        M3_refl = np.ones_like(self.lam.value)

        # M4 reflectivity
        print("Warning: M4 reflectivity not included in YAML")
        M4_refl = np.ones_like(self.lam.value)

        total_tele_refl = M1_refl * M2_refl * M3_refl * M4_refl

        print("Done.")

        # save parameters as class properties
        self.diam_insc = diam_insc
        self.diam_circ = diam_circ
        self.M1_refl = M1_refl
        self.M2_refl = M2_refl
        self.M3_refl = M3_refl
        self.M4_refl = M4_refl
        self.total_tele_refl = total_tele_refl

    def load_EAC2(self):
        eac2_fl = "/obs_config/Tel/EAC2.yaml"
        print(f"Loading file: {eac2_fl}")
        eac2_dict = load_yaml(eac2_fl)
        print(eac2_dict)
        print("EAC2 LOAD SCRIPT NOT YET IMPLEMENTED")

    def load_EAC3(self):
        eac3_fl = "/obs_config/Tel/EAC3.yaml"
        print(f"Loading file: {eac3_fl}")
        eac3_dict = load_yaml(eac3_fl)
        print(eac3_dict)
        print("EAC3 LOAD SCRIPT NOT YET IMPLEMENTED")

    def load_custom(self, diam_insc, diam_circ, total_tele_refl):
        self.diam_insc = diam_insc
        self.diam_circ = diam_circ
        self.total_tele_refl = total_tele_refl


class CI:
    def __init__(self, lam):
        print("Initializing Coronagraph Instrument")
        self.lam = lam  # the native wavelength grid we are working in (in um)
        # self.total_inst_refl = total_inst_refl
        # self.TCA_refl = TCA_refl
        # self.wb_tran = wb_tran
        # self.wb_refl = wb_refl
        # self.pb_refl = pb_refl
        # self.FSM_refl = FSM_refl
        # self.OAPsf_refl = OAPsf_refl
        # self.DM1_refl = DM1_refl
        # self.DM2_refl = DM2_refl
        # self.Fold_refl = Fold_refl
        # self.OAPsb_refl = OAPsb_refl
        # self.Apodizer_refl = Apodizer_refl
        # self.FPM_refl = FPM_refl
        # self.Lyot_refl = Lyot_refl
        # self.FStop_refl = FStop_refl
        # self.filters_refl = filters_refl

    def plot(self):

        fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
        try:
            axes[0].plot(self.lam, self.total_inst_refl, label="Total instrument refl")
        except AttributeError:
            print("####################################################")
            print("Nothing to plot. Try running CI.calculate_throughput")
            print("####################################################")

        axes[1].plot(self.lam, self.TCA_refl, label="TCA")
        axes[1].plot(self.lam, self.wb_tran, label="wave_beamsplitter tran")
        axes[1].plot(self.lam, self.wb_refl, label="wave_beamsplitter refl")
        axes[1].plot(self.lam, self.pb_refl, label="pol_beamsplitter")
        axes[1].plot(self.lam, self.FSM_refl, label="FSM")
        axes[1].plot(self.lam, self.OAPsf_refl, label="OAPs_forward")
        axes[1].plot(self.lam, self.DM1_refl, label="DM1")
        axes[1].plot(self.lam, self.DM2_refl, label="DM2")
        axes[1].plot(self.lam, self.Fold_refl, label="Fold")
        axes[1].plot(self.lam, self.OAPsb_refl, label="OAPs_back")
        axes[1].plot(self.lam, self.Apodizer_refl, label="Apodizer")
        axes[1].plot(self.lam, self.FPM_refl, label="Focal_Plane_Mask")
        axes[1].plot(self.lam, self.Lyot_refl, label="Lyot_Stop")
        axes[1].plot(self.lam, self.FStop_refl, label="Field_Stop")
        axes[1].plot(self.lam, self.filters, label="filters")
        axes[1].set_xlabel("Wavelength [um]")
        axes[0].set_ylabel("Trans/refl")
        axes[1].legend()
        plt.show()

    def calculate_throughput(self):
        ci_fl = "/obs_config/CI/CI.yaml"
        print(f"Loading file: {ci_fl}\n")

        # Full optical Path: 'PM','SM','TCA','TCA','TCA','TCA','wave_beamsplitter', 'pol_beamsplitter', 'FSM', 'OAPs_forward', 'OAPs_forward', 'DM1', 'DM2', 'OAPs_forward', 'Fold', 'OAPs_back', 'Apodizer', 'OAPs_back', 'Focal_Plane_Mask', 'OAPs_back', 'Lyot_Stop', 'OAPs_back', 'Field_Stop', 'OAPs_back', 'filters', 'OAPs_back', 'Detector'
        ci_dict = load_yaml(ci_fl)
        print("Optical path:")
        print(ci_dict["opticalpath"]["full_path"])

        print("Calculating throughput...")

        # TCA
        TCA_reflectivity_fl = ci_dict["TCA"]["reflectivity"]
        TCA_reflectivity_dict = load_yaml(
            reflectivity_path + TCA_reflectivity_fl.split("/")[-1]
        )
        TCA_refl = TCA_reflectivity_dict["reflectivity"]
        TCA_lam = TCA_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        TCA_refl = interp_arr(TCA_lam, TCA_refl, self.lam)

        # wave_beamsplitter
        wb_reflectivity_fl = ci_dict["wave_beamsplitter"]["reflectivity"]  # < 1 um
        wb_transmission_fl = ci_dict["wave_beamsplitter"]["transmission"]  # > 1 um
        wb_reflectivity_dict = load_yaml(
            reflectivity_path + wb_reflectivity_fl.split("/")[-1]
        )
        wb_transmission_dict = load_yaml(
            reflectivity_path + wb_transmission_fl.split("/")[-1]
        )
        wb_refl = wb_reflectivity_dict["reflectivity"]
        wb_refl_lam = wb_reflectivity_dict["wavelength"] * u.nm
        wb_tran = wb_transmission_dict["reflectivity"]
        wb_tran_lam = wb_transmission_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        wb_refl = interp_arr(wb_refl_lam, wb_refl, self.lam)
        wb_tran = interp_arr(wb_tran_lam, wb_tran, self.lam)

        # pol_beamsplitter
        # no transmission/reflectivity profiles here
        pb_refl = np.ones_like(self.lam.value)

        # FSM
        FSM_reflectivity_fl = ci_dict["FSM"]["reflectivity"]
        FSM_reflectivity_dict = load_yaml(
            reflectivity_path + FSM_reflectivity_fl.split("/")[-1]
        )
        FSM_refl = FSM_reflectivity_dict["reflectivity"]
        FSM_lam = FSM_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        FSM_refl = interp_arr(FSM_lam, FSM_refl, self.lam)

        # OAPs_forward
        OAPsf_reflectivity_fl = ci_dict["OAPs_forward"]["reflectivity"]
        OAPsf_reflectivity_dict = load_yaml(
            reflectivity_path + OAPsf_reflectivity_fl.split("/")[-1]
        )
        OAPsf_refl = OAPsf_reflectivity_dict["reflectivity"]
        OAPsf_lam = OAPsf_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        OAPsf_refl = interp_arr(OAPsf_lam, OAPsf_refl, self.lam)

        # DM1
        DM1_reflectivity_fl = ci_dict["DM1"]["reflectivity"]
        DM1_reflectivity_dict = load_yaml(
            reflectivity_path + DM1_reflectivity_fl.split("/")[-1]
        )
        DM1_refl = DM1_reflectivity_dict["reflectivity"]
        DM1_lam = DM1_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        DM1_refl = interp_arr(DM1_lam, DM1_refl, self.lam)

        # DM2
        DM2_reflectivity_fl = ci_dict["DM2"]["reflectivity"]
        DM2_reflectivity_dict = load_yaml(
            reflectivity_path + DM2_reflectivity_fl.split("/")[-1]
        )
        DM2_refl = DM2_reflectivity_dict["reflectivity"]
        DM2_lam = DM2_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        DM2_refl = interp_arr(DM2_lam, DM2_refl, self.lam)

        # Fold
        Fold_reflectivity_fl = ci_dict["Fold"]["reflectivity"]
        Fold_reflectivity_dict = load_yaml(
            reflectivity_path + Fold_reflectivity_fl.split("/")[-1]
        )
        Fold_refl = Fold_reflectivity_dict["reflectivity"]
        Fold_lam = Fold_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        Fold_refl = interp_arr(Fold_lam, Fold_refl, self.lam)

        # OAPs_back
        OAPsb_reflectivity_fl = ci_dict["OAPs_back"]["reflectivity"]
        OAPsb_reflectivity_dict = load_yaml(
            reflectivity_path + OAPsb_reflectivity_fl.split("/")[-1]
        )
        OAPsb_refl = OAPsb_reflectivity_dict["reflectivity"]
        OAPsb_lam = OAPsb_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        OAPsb_refl = interp_arr(OAPsb_lam, OAPsb_refl, self.lam)

        # Apodizer
        Apodizer_reflectivity_fl = ci_dict["Apodizer"]["reflectivity"]
        Apodizer_reflectivity_dict = load_yaml(
            reflectivity_path + Apodizer_reflectivity_fl.split("/")[-1]
        )
        Apodizer_refl = Apodizer_reflectivity_dict["reflectivity"]
        Apodizer_lam = Apodizer_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        Apodizer_refl = interp_arr(Apodizer_lam, Apodizer_refl, self.lam)

        # Focal_Plane_Mask
        FPM_reflectivity_fl = ci_dict["Focal_Plane_Mask"]["transmission"]
        FPM_reflectivity_dict = load_yaml(
            reflectivity_path + FPM_reflectivity_fl.split("/")[-1]
        )
        FPM_refl = FPM_reflectivity_dict["reflectivity"]
        FPM_lam = FPM_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        FPM_refl = interp_arr(FPM_lam, FPM_refl, self.lam)

        # Lyot_Stop
        Lyot_reflectivity_fl = ci_dict["Lyot_Stop"]["reflectivity"]
        Lyot_reflectivity_dict = load_yaml(
            reflectivity_path + Lyot_reflectivity_fl.split("/")[-1]
        )
        Lyot_refl = Lyot_reflectivity_dict["reflectivity"]
        Lyot_lam = Lyot_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        Lyot_refl = interp_arr(Lyot_lam, Lyot_refl, self.lam)

        # Field_Stop
        FStop_reflectivity_fl = ci_dict["Field_Stop"]["transmission"]
        FStop_reflectivity_dict = load_yaml(
            reflectivity_path + FStop_reflectivity_fl.split("/")[-1]
        )
        FStop_refl = FStop_reflectivity_dict["reflectivity"]
        FStop_lam = FStop_reflectivity_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        FStop_refl = interp_arr(FStop_lam, FStop_refl, self.lam)

        # filters
        # Filters not implemented yet
        filters = np.ones_like(self.lam.value)

        # for total reflectivity/transmission, follow the optical path:
        # Full optical Path: 'PM','SM','TCA','TCA','TCA','TCA','wave_beamsplitter', 'pol_beamsplitter', 'FSM', 'OAPs_forward', 'OAPs_forward', 'DM1', 'DM2', 'OAPs_forward', 'Fold', 'OAPs_back', 'Apodizer', 'OAPs_back', 'Focal_Plane_Mask', 'OAPs_back', 'Lyot_Stop', 'OAPs_back', 'Field_Stop', 'OAPs_back', 'filters', 'OAPs_back', 'Detector'
        total_inst_refl = (
            TCA_refl
            * TCA_refl
            * TCA_refl
            * TCA_refl
            * (wb_tran + wb_refl)
            * pb_refl
            * FSM_refl
            * OAPsf_refl
            * OAPsf_refl
            * DM1_refl
            * DM2_refl
            * OAPsf_refl
            * Fold_refl
            * OAPsb_refl
            * Apodizer_refl
            * OAPsb_refl
            * FPM_refl
            * OAPsb_refl
            * Lyot_refl
            * OAPsb_refl
            * FStop_refl
            * OAPsb_refl
            * filters
            * OAPsb_refl
        )

        # save parameters as class properties
        self.total_inst_refl = total_inst_refl
        self.TCA_refl = TCA_refl
        self.wb_tran = wb_tran
        self.wb_refl = wb_refl
        self.pb_refl = pb_refl
        self.FSM_refl = FSM_refl
        self.OAPsf_refl = OAPsf_refl
        self.DM1_refl = DM1_refl
        self.DM2_refl = DM2_refl
        self.Fold_refl = Fold_refl
        self.OAPsb_refl = OAPsb_refl
        self.Apodizer_refl = Apodizer_refl
        self.FPM_refl = FPM_refl
        self.Lyot_refl = Lyot_refl
        self.FStop_refl = FStop_refl
        self.filters = filters

        print("Done.")

        # ci = CI(lam, total_inst_refl, TCA_refl, wb_tran, wb_refl, pb_refl, FSM_refl, OAPsf_refl, DM1_refl, DM2_refl, Fold_refl, OAPsb_refl, Apodizer_refl, FPM_refl, Lyot_refl, FStop_refl, filters)


class DETECTOR:
    def __init__(self, lam):
        self.lam = lam

    def load_imager(self):

        print("Loading Broadband Imager...")

        ci_dict = load_yaml("/obs_config/CI/CI.yaml")

        # visible channels
        vis_imager = ci_dict["Visible_Channels"]["Detectors"]["Broadband_Imager"]
        qe_vis_fl = vis_imager["QE"]
        qe_vis_dict = load_yaml(detectors_path + qe_vis_fl.split("/")[-1])
        qe_vis = qe_vis_dict["QE"]
        qe_vis_lam = qe_vis_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        qe_vis = interp_arr(qe_vis_lam, qe_vis, self.lam)

        rn_vis = vis_imager["RN"][0]  # electrons/pix      * u.Unit(vis_imager["RN"][1])
        dc_vis = vis_imager["DC"][0]  # electrons/pix/s    * u.Unit(vis_imager["DC"][1])
        cic_vis = None  # NOT IMPLEMENTED YET

        # save parameters as class properties
        self.qe_vis = qe_vis
        self.rn_vis = rn_vis
        self.dc_vis = dc_vis
        self.cic_vis = cic_vis

        # nir channels
        nir_imager = ci_dict["NIR_Channels"]["Detectors"]["Broadband_Imager"]
        qe_nir_fl = nir_imager["QE"]
        qe_nir_dict = load_yaml(detectors_path + qe_nir_fl.split("/")[-1])
        qe_nir = qe_nir_dict["QE"]
        qe_nir_lam = qe_nir_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        qe_nir = interp_arr(qe_nir_lam, qe_nir, self.lam)

        rn_nir = nir_imager["RN"][
            0
        ]  # electrons/pix        * u.Unit(nir_imager["RN"][1])
        dc_nir = nir_imager["DC"][
            0
        ]  # electrons/pix/s       * u.Unit(nir_imager["DC"][1])
        cic_nir = None  # NOT IMPLEMENTED YET

        # save parameters as class properties
        self.qe_nir = qe_nir
        self.rn_nir = rn_nir
        self.dc_nir = dc_nir
        self.cic_nir = cic_nir

    def load_IFS(self):
        print("Loading IFS...")

        ci_dict = load_yaml("/obs_config/CI/CI.yaml")

        # visible channels
        vis_ifs = ci_dict["Visible_Channels"]["Detectors"]["IFS"]
        qe_vis_fl = vis_ifs["QE"]
        qe_vis_dict = load_yaml(detectors_path + qe_vis_fl.split("/")[-1])
        qe_vis = qe_vis_dict["QE"]
        print(qe_vis)
        qe_vis_lam = qe_vis_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        qe_vis = interp_arr(qe_vis_lam, qe_vis, self.lam)
        print(qe_vis)

        rn_vis = vis_ifs["RN"][0]  # electrons/pix
        dc_vis = vis_ifs["DC"][0]  # electrons/pix/s
        cic_vis = None  # NOT IMPLEMENTED YET

        # save parameters as class properties
        self.qe_vis = qe_vis
        self.rn_vis = rn_vis
        self.dc_vis = dc_vis
        self.cic_vis = cic_vis

        # nir channels
        nir_ifs = ci_dict["NIR_Channels"]["Detectors"]["IFS"]
        qe_nir_fl = nir_ifs["QE"]
        qe_nir_dict = load_yaml(detectors_path + qe_nir_fl.split("/")[-1])
        qe_nir = qe_nir_dict["QE"]
        qe_nir_lam = qe_nir_dict["wavelength"] * u.nm
        # interpolate onto our wavelength grid
        qe_nir = interp_arr(qe_nir_lam, qe_nir, self.lam)

        rn_nir = nir_ifs["RN"][0]  # electrons/pix
        dc_nir = nir_ifs["DC"][0]  # electrons/pix/s
        cic_nir = None  # NOT IMPLEMENTED YET

        # save parameters as class properties
        self.qe_nir = qe_nir
        self.rn_nir = rn_nir
        self.dc_nir = dc_nir
        self.cic_nir = cic_nir

    def plot(self):

        plt.figure()
        plt.plot(self.lam, self.qe_vis, label="QE VIS")
        plt.plot(self.lam, self.qe_nir, label="QE NIR")
        plt.xlabel("Wavelength [um]")
        plt.ylabel("Quantum Efficiency")
        plt.legend()
        plt.show()
        print("VISIBLE")
        print("-------")
        print(f"RN: {self.rn_vis}")
        print(f"DC: {self.dc_vis}")
        print("\n")
        print("NIR")
        print("---")
        print(f"RN: {self.rn_nir}")
        print(f"DC: {self.dc_nir}")
