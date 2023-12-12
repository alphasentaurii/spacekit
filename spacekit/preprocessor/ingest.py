import os
import sys
import glob
import shutil
import pandas as pd
from spacekit.logger.log import Logger
from spacekit.extractor.scrape import JsonScraper
from spacekit.preprocessor.scrub import HstSvmScrubber, JwstCalScrubber
from spacekit.generator.draw import DrawMosaics
from spacekit.analyzer.track import timer, record_metrics
from spacekit.skopes.jwst.cal.config import KEYPAIR_DATA
from spacekit.extractor.radio import JwstCalRadio



class SvmAlignmentIngest:
    def __init__(self, input_path, batch_out):
        self.input_path = input_path
        self.batch_out = os.getcwd() if batch_out is None else batch_out
        self.log_dir = None
        self.clean = True
        self.visit_data = []
        self.data_paths = []
        self.json_pattern = "*_total*_svm_*.json"
        self.crpt = 0
        self.draw = 1
        self.img_outputs = os.path.join(self.batch_out, "img")

    def start(self, func, ps="prep", visit=None, **args):
        t0, start = timer()
        func.__call__(**args)
        wall, clock = timer(t0=t0, clock=start)
        record_metrics(self.log_dir, visit, wall, clock, ps=ps)

    def prep_svm_batch(self, batch_name=None, drz_ver="3.4.1"):
        batch_name = "drz" if batch_name is None else batch_name
        if drz_ver:
            drz = "".join(drz_ver.split("."))
            batch_name += f"_{drz}"
        self.run_preprocessing(self, fname=batch_name, output_path=self.batch_out)

    def prep_single_visit(self, visit_path):
        visit = str(os.path.basename(visit_path))
        drz_file = glob.glob(f"{visit_path}/*total*.fits")
        if len(drz_file) > 0:
            dets = [drz.split("/")[-1].split("_")[4] for drz in drz_file]
            try:
                input_path = os.path.dirname(visit_path)
                for det in dets:
                    _, _ = self.run_preprocessing(
                        input_path,
                        fname=f"{visit}_{det.lower()}_data",
                        output_path=self.batch_out,
                        visit=visit,
                    )
                if self.clean is True:
                    shutil.rmtree(visit_path)
            except Exception as e:
                print(e)
                sys.exit(1)

    def run_preprocessing(
        self,
        h5=None,
        fname="svm_data",
        visit=None,
    ):
        """Scrapes SVM data from raw files, preprocesses dataframe for MLP classifier and generates png images for image CNN.
        #TODO: if no JSON files found, look for results_*.csv file instead and preprocess via alternative method

        Parameters
        ----------
        input_path : str
            path to SVM dataset directory
        h5 : str, optional
            load from existing hdf5 file, by default None
        fname : str, optional
            base filename to give the output files, by default "svm_data"
        output_path : str, optional
            where to save output files. Defaults to current working directory, by default None
        json_pattern : str, optional
            glob-based search pattern, by default "*_total*_svm_*.json"
        visit: str, optional
            single visit name (e.g. "id8f34") matching subdirectory of input_path; will search and preprocess this visit only
            (rather than all visits contained in the input_path), by default None
        crpt : int, optional
            set to 1 if using synthetic corruption data, by default 0
        draw : int, optional
            generate png images from dataset, by default 1

        Returns
        -------
        dataframe
            preprocessed Pandas dataframe
        """
        os.makedirs(self.batch_out, exist_ok=True)
        fname = os.path.basename(fname).split(".")[0]
        # 1: SCRAPE JSON FILES and make dataframe
        if h5 is None:
            search_path = (
                os.path.join(self.input_path, visit) if visit else self.input_path
            )
            patterns = self.json_pattern.split(",")
            jsc = JsonScraper(
                search_path=search_path,
                search_patterns=patterns,
                file_basename=fname,
                crpt=self.crpt,
                output_path=self.batch_out,
            )
            jsc.json_harvester()
        else:
            jsc = JsonScraper(h5_file=h5).load_h5_file()
        # 2: Scrape Fits Files and SCRUB DATAFRAME
        scrub = HstSvmScrubber(
            self.input_path,
            data=jsc.data,
            output_path=self.batch_out,
            output_file=fname,
            crpt=self.crpt,
        )
        scrub.preprocess_data()
        # 3:  DRAW IMAGES
        if self.draw:
            img_outputs = os.path.join(self.batch_out, "img")
            mos = DrawMosaics(
                self.input_path,
                output_path=img_outputs,
                fname=scrub.data_path,
                pattern="",
                gen=3,
                size=(24, 24),
                crpt=self.crpt,
            )
            mos.generate_total_images()

        self.visit_data.append(scrub.df)
        self.data_paths.append(scrub.data_path)
        print(f"DATA PATH: {scrub.data_path}\n")
        print(scrub.df_visit)
        return scrub.df, scrub.data_path

    def concat_prepped(dpath):
        datafiles = glob.glob(f"{dpath}/??????_*data.csv")
        for i, fpath in enumerate(datafiles):
            df_visit = pd.read_csv(fpath, index_col="index")
            if i == 0:
                df = df_visit
            else:
                df = pd.concat([df, df_visit], axis=0)
        if "index" not in df.columns:
            df["index"] = df.index
        df.to_csv(f"{dpath}/preprocessed.csv", index=False)

    def concat_raw(dpath):
        rawfiles = glob.glob(f"{dpath}/raw_*_data.csv")
        for i, raw in enumerate(rawfiles):
            df_raw = pd.read_csv(raw, index_col="index")
            if i == 0:
                df = df_raw
            else:
                df = pd.concat([df, df_raw], axis=0)
        if "index" not in df.columns:
            df["index"] = df.index
        df.to_csv(f"{dpath}/raw_combined.csv", index=False)

    def final_cleanup(df, dpath):
        print("Cleaning up...")
        csvfiles = glob.glob(f"{dpath}/*_data.csv")
        h5files = glob.glob(f"{dpath}/*_data.h5")
        rawfiles = glob.glob(f"{dpath}/raw_*_data.csv")
        filegroups = [csvfiles, h5files, rawfiles]
        for grp in filegroups:
            if len(df) == len(grp):
                for f in grp:
                    os.remove(f)
                print(f"Cleaned up {len(grp)} files")
            else:
                print(
                    f"{len(df)} in DF does not match {len(grp)} in filegroup. Skipping cleanup"
                )




class JwstCalIngest:
    def __init__(self, input_path=None, pfx="", sfx=".csv", name="JwstCalIngest", **log_kws):
        self.input_path = input_path
        self.pfx = pfx
        self.sfx = sfx
        self.files = self.read_files()
        self.index = "Dataset"
        self.df = None
        self.__name__ = name
        self.log = Logger(self.__name__, **log_kws).spacekit_logger()

    def extract_pid(self, x):
        if not isinstance(x, str):
            return x
        pid = x[2:7]
        if pid[0] == '0':
            pid = pid[1:]
        return int(pid)

    def convert_to_float(self, x):
        if x != "NONE":
            return float(x)
    
    def read_files(self):
        if self.input_path is None:
            self.input_path = os.getcwd()
        self.files = glob.glob(f"{self.input_path}/{self.pfx}*{self.sfx}")
        # ['2023-11-23_327.csv',
        # '2023-11-28_332.csv',
        # '2023-11-24_328.csv',
        # '2023-11-30_334.csv',
        # '2023-11-29_333.csv',
        # '2023-11-25_329.csv',
        # '2023-11-26_330.csv',
        # '2023-11-27_331.csv']

    def ingest_data(self):
        for f in self.files:
            df = pd.read_csv(f, index_col=self.index)
            if self.df is None:
                self.df = df
            else:
                self.df = pd.concat([self.df, df])
    
    def scrub_exposures(self):
        self.df['dname'] = self.df.index
        self.df['pid'] = self.df['dname'].apply(lambda x: self.extract_pid(x))
        float_cols = [
            'CRVAL1',
            'CRVAL2',
            'RA_REF',
            'DEC_REF',
            'GS_RA',
            'GS_DEC',
            'TARG_RA',
            'TARG_DEC'
        ]
        for col in float_cols:
            self.df[col] = self.df[col].apply(lambda x: self.convert_to_float(x))
        radio = JwstCalRadio()
        # IMAGE DATA
        IMAGE_EXPTYPES = ['NRC_IMAGE', 'MIR_IMAGE', 'NIS_IMAGE']
        # nrcimg_l1 = self.df.loc[(self.df.DagNodeName=='LEVEL_1')&(self.df.EXP_TYPE=='NRC_IMAGE')]
        # mirimg_l1 = self.df.loc[(self.df.DagNodeName=="LEVEL_1")&(self.df.EXP_TYPE=="MIR_IMAGE")]
        # nisimg_l1 =  self.df.loc[(self.df.DagNodeName=="LEVEL_1")&(self.df.EXP_TYPE=="NIS_IMAGE")]
        for exp_type in IMAGE_EXPTYPES:
            data = self.df.loc[(self.df.DagNodeName=='LEVEL_1')&(self.df.EXP_TYPE==exp_type)]
            scrubber = JwstCalScrubber(
                self.input_path,
                data=data,
                encoding_pairs=KEYPAIR_DATA,
                mode='df'
            )
            self.df = radio.match_asn_filename(self.df, scrubber.img_products)
        product_list = list(
            self.df.loc[(
                self.df['product'] != "NONE"
                ) & (
                    self.df['product'].isna() == False
                )]['product'].unique()
        )
        import re
        pattern = re.compile(r"jw[0-9]{5}-o[0-9]{3}_t[0-9]{3}_*") 
        l3_prods = []
        for p in product_list:
            m = re.match(pattern, p)
            if m:
                l3_prods.append(p)

        # nircam = self.df.loc[(df['product'].isin(l3_prods)) & (df['INSTRUME']=="NIRCAM") & (df.DagNodeName!='LEVEL_2A')]
        # mirimage =  df.loc[(df['product'].isin(l3_prods))& (df.DagNodeName!='LEVEL_2A')]