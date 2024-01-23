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
        self.exp_types = ["IMAGE", "SPEC", "TAC"]
        self.files = None
        self.index = "Dataset"
        self.dagcol = "DagNodeName"
        self.df = None
        self.l1_dags = None
        self.input_dict = None
        self.input_data = {}
        self.product_matches = None
        self.unmatched = {}
        # *** TEMP *** #
        self.asn_mode = False
        self.scrubber = None
        self.__name__ = name
        self.log = Logger(self.__name__, **log_kws).spacekit_logger()
    
    def run_ingest(self, outpath=None, file_suffix="-2023"):
        self.read_files()
        if len(self.files) == 0:
            return
        self.ingest_data()
        self.scrub_exposures()
        self.save_training_sets(self, outpath=outpath, sfx=file_suffix)

    def strip_file_suffix(self, x):
        if x.endswith("fits"):
            x = '_'.join(x.split('_')[:-1])
        return x

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
        self.files = []
        pattern = f"{self.input_path}/{self.pfx}*{self.sfx}"
        self.files = sorted(glob.glob(pattern))
        if len(self.files) < 1:
            self.log.warning(f"No files found using pattern: {pattern}")

    def drop_l2_data(self, df):
        alldags = sorted(list(df[self.dagcol].value_counts().index))
        self.l1_dags = [d for d in alldags if '1' in d]
        dags_l1_l3 =  [d for d in alldags if d in self.l1_dags or '3' in d]
        df = df.loc[df[self.dagcol].isin(dags_l1_l3)]
        return df

    def ingest_data(self):
        for f in self.files:
            df = pd.read_csv(f, index_col=self.index)
            df = self.drop_l2_data(df)
            if self.df is None:
                self.df = df
            else:
                self.df = pd.concat([self.df, df], axis=0)

    def scrub_exposures(self):
        self.df['dname'] = self.df.index
        self.df['dname'] = self.df['dname'].apply(lambda x: self.strip_file_suffix(x))
        self.df.set_index('dname', drop=False, inplace=True)
        self.df['pid'] = self.df['dname'].apply(lambda x: self.extract_pid(x))
        self.df.rename({'TARGNAME':'targname', 'ImageSize':'imagesize'}, axis=1, inplace=True)
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
        
        self.scrubber = JwstCalScrubber(
                self.input_path,
                data=self.df.loc[self.df[self.dagcol].isin(self.l1_dags)],
                encoding_pairs=KEYPAIR_DATA,
                mode='df'
            )
        self.input_dict = self.scrubber.input_data()

        for exp_type in self.exp_types:
            inputs = self.scrubber.scrub_inputs(exp_type=exp_type)
            if inputs is not None:
                inputs['dname'] = inputs.index
                self.input_data[exp_type] = inputs
        # *** TEMP *** #
        if self.asn_mode is True:
            self.match_asn_products(self.scrubber)
        else:
            for exp_type, products in list(zip(["IMAGE", "SPEC", "TAC"], [
                self.scrubber.img_products,
                self.scrubber.spec_products,
                self.scrubber.tac_products
                ])):
                self.match_product_groups(products, exp_type=exp_type)
        self.drop_incomplete_data()
        self.convert_imagesize_units()

    def match_product_groups(self, products, exp_type="IMAGE"):
        for k, v in products.items():
            exposure = list(v.keys())[0]
            info = self.df.loc[exposure]
            if isinstance(info, pd.DataFrame):
                if 'LEVEL_1' in list(info['DagNodeName'].unique()):
                    info = info.loc[info['DagNodeName'] == 'LEVEL_1'].iloc[0]
                else:
                    info = info.iloc[0]
            l3 = self.df.loc[
                (
                    self.df['pid'] == info['pid']
                ) & (
                    self.df['OBSERVTN'] == info['OBSERVTN']
                ) & (
                    self.df['FILTER'] == info['FILTER']
                ) & (
                    self.df['GRATING'] == info['GRATING']
                ) & (
                    self.df['PUPIL'] == info['PUPIL']
                ) & (
                    self.df['EXP_TYPE'] == info['EXP_TYPE']
                ) & (
                    ~self.df['DagNodeName'].isin(self.l1_dags)
                )
            ]
            if len(l3) == 0:
                self.log.debug(f"No matching products identified: {k}")
                continue
            elif len(l3) > 1:
                self.log.debug(f"Multiple products match: {k}")
                pnames = sorted(list(l3.index), reverse=True)
                pname = pnames[0] # default if better match not found
                prefix = k.split('_')[0]
                for p in pnames:
                    if p.split('_')[0] == prefix:
                        pname = p
                        break
                    else:
                        continue
                imagesize = l3.loc[pname]['imagesize']
                if not isinstance(imagesize, int):
                    imagesize = imagesize.max()
            else:
                pname = l3.dname[0]
                imagesize = l3.imagesize[0]
            self.input_data[exp_type].loc[k, 'pname'] = pname
            self.input_data[exp_type].loc[k, 'imagesize'] = imagesize
            self.df.loc[pname, 'pname'] = pname
            for e in list(v.keys()):
                self.df.loc[e, 'pname'] = pname

    def match_asn_products(self, scrubber):
        radio = JwstCalRadio()
        self.product_matches = radio.match_asn_filename(self.input_data)
        for exp_type, products in list(zip(["IMAGE", "SPEC", "TAC"], [
            scrubber.img_products,
            scrubber.spec_products,
            scrubber.tac_products
            ])):
            for k, v in products.items():
                info = self.product_matches[exp_type].get(k, None)
                if info:
                    try:
                        imagesize = self.df.loc[self.df.dname == info['asn']]['imagesize'][0]
                    except IndexError:
                        imagesize = None # no L3 product
                    for key, value in info.items():
                        self.input_data[exp_type].loc[k, key] = value
                        for e in list(v.keys()):
                            self.df.loc[e, key] = value

                    if imagesize is not None:
                        self.df.loc[info['asn'], 'pname'] = info['pname']
                        self.df.loc[info['asn'], 'targname'] = info['targname']
                        self.input_data[exp_type].loc[k, 'imagesize'] = imagesize

    def drop_incomplete_data(self):
        for exp_type in self.exp_types:
            if 'imagesize' in self.input_data[exp_type].columns:
                self.unmatched[exp_type] = self.input_data[exp_type].loc[self.input_data[exp_type]['imagesize'].isna()==True].index
            else:
                self.unmatched[exp_type] = self.input_data[exp_type].index
            self.log.info(f"Dropping {len(self.unmatched[exp_type])} incomplete inputs for {exp_type}")
            self.input_data[exp_type].drop(self.unmatched[exp_type], axis=0, inplace=True)

    def convert_imagesize_units(self):
        for exp_type in self.exp_types:
            if 'imagesize' in self.input_data[exp_type].columns:
                self.input_data[exp_type]['imgsize_gb'] = self.input_data[exp_type]['imagesize'].apply(lambda x: x // 10**6)
    
    def save_training_sets(self, outpath=None, sfx="-2023"):
        for exp_type in self.exp_types:
            data = self.input_data[exp_type].copy()
            data[self.index] = data.index
            if outpath is None:
                outpath = self.input_path
            data.to_csv(f"{outpath}/train-{exp_type.lower()}{sfx}.csv", index=False)
