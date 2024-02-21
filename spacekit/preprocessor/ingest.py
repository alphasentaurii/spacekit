import os
import sys
import glob
import shutil
import pandas as pd
from argparse import ArgumentParser
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
    def __init__(self, input_path=None, pfx="", outpath=None, batch_id=None, name="JwstCalIngest", **log_kws):
        self.input_path = input_path
        self.pfx = pfx
        self.outpath = input_path if outpath is None else outpath
        self.batch_id = batch_id
        self.exp_types = ["IMAGE", "SPEC", "TAC"]
        self.files = []
        self.idxcol = "Dataset"
        self.dagcol = "DagNodeName"
        self.df = None
        self.l1_dags = []
        self.l3_dags = []
        self.data = {}
        self.product_matches = None
        self.rem = {}
        self.param_cols = ['pid', 'OBSERVTN', 'FILTER', 'GRATING', 'PUPIL', 'EXP_TYPE']
        self.scrb = None
        self.__name__ = name
        self.log = Logger(self.__name__, **log_kws).spacekit_logger()
        # TEMP #
        self.log.console_log_level = "DEBUG"

    def run_ingest(self, extrapolate=True, file_suffix=""):
        self.ingest_data()
        if len(self.files) == 0:
            return
        self.initial_scrub()
        self.scrub_exposures()
        self.run_matching()
        self.drop_incomplete_data()
        self.convert_imagesize_units()
        if extrapolate is True:
            self.extrapolate_datasets(fpath=None)
        self.save_training_sets()
    
    def read_files(self):
        if self.input_path is None:
            self.input_path = os.getcwd()
        pattern = f"{self.input_path}/{self.pfx}*.csv"
        files = sorted(glob.glob(pattern))
        self.files = [f for f in files if f not in self.files]
        if len(self.files) < 1:
            self.log.warning(f"No files found using pattern: {pattern}")
        else:
            self.log.debug(f"Files ready for ingest: {self.files}")

    def drop_level2(self, df):
        alldags = sorted(list(df[self.dagcol].value_counts().index))
        l1_dags = [d for d in alldags if '1' in d]
        l3_dags = [d for d in alldags if '3' in d]
        dags_l1_l3 = l1_dags + l3_dags
        df = df.loc[df[self.dagcol].isin(dags_l1_l3)]
        self.l1_dags.extend([l for l in l1_dags if l not in self.l1_dags])
        self.l3_dags.extend([l for l in l3_dags if l not in self.l3_dags])
        return df

    def ingest_data(self):
        self.read_files()
        for f in self.files:
            df = pd.read_csv(f, index_col=self.idxcol)
            df = self.drop_level2(df)
            filedate, day = os.path.basename(f).split('_')
            df['date'] = filedate
            df['year'] = filedate.split('-')[0]
            df['doy'] = int(day.split('.')[0])
            if self.df is None:
                self.df = df
            else:
                self.df = pd.concat([self.df, df], axis=0)

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

    # def validate_obs(self, x):
    #     if len(str(x)) <= 3:
    #         return '{:0>3}'.format(x)

    def mark_mosaics(self, x):
        if len(x.split('-')) < 2:
            return False
        elif x.split('-')[1][0] != 'c':
            return False
        return True

    def drop_duplicates(self, priority='imagesize'):
        if priority is None:
            self.df.drop_duplicates(subset='dname', inplace=True)
        dupes = sorted(list(self.df.loc[self.df.duplicated(subset='dname')].index))
        self.df.loc[dupes, 'dupe'] = True
        self.df.loc[self.df.dupe.isna(), 'dupe'] = False
        for d in dupes:
            imgmax = self.df.loc[d]['imagesize'].max()
            self.df.loc[(self.df.dname == d) & (self.df['imagesize'] == imgmax), 'dupe'] = False
        self.df.reset_index(inplace=True)
        self.log.info(f"Dropping {len(dupes)} duplicates (priority=imagesize)")
        self.df.drop(self.df.loc[self.df.dupe].index, axis=0, inplace=True)
        self.df.drop('dupe', axis=1, inplace=True)
        if len(sorted(list(self.df.loc[self.df.duplicated(subset='dname')].index))) > 0:
            self.df.drop_duplicates(subset='dname', inplace=True)

    def extract_obs(self, x):
        pfx = x.split('_')[0]
        pfx2 = pfx.split('-')
        if len(pfx2) > 1:
            obs = pfx2[1][1:]
        else:
            obs = pfx[7:10]
        return obs

    def initial_scrub(self):
        self.df['dname'] = self.df.index
        self.df['dname'] = self.df['dname'].apply(lambda x: self.strip_file_suffix(x))
        self.df.rename({'ImageSize':'imagesize'}, axis=1, inplace=True)
        self.drop_duplicates()
        self.df.set_index('dname', drop=False, inplace=True)
        self.df['pid'] = self.df['dname'].apply(lambda x: self.extract_pid(x))
        self.df['OBSERVTN'] = self.df['OBSERVTN'].apply(lambda x: '{:0>3}'.format(x))
        params = list(map(lambda x: '-'.join([str(y) for y in x if y != "NONE"]),  self.df[self.param_cols].values))
        self.df['params'] = pd.DataFrame(params, index=self.df.index)
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
        self.drop_mosaics()


    def scrub_exposures(self):
        self.scrb = JwstCalScrubber(
                self.input_path,
                data=self.df.loc[self.df[self.dagcol].isin(self.l1_dags)],
                encoding_pairs=KEYPAIR_DATA,
                mode='df'
            )
        for exp_type in self.exp_types:
            inputs = self.scrb.scrub_inputs(exp_type=exp_type)
            if inputs is not None:
                inputs['dname'] = inputs.index
                self.data[exp_type] = inputs
        (self.img, self.spec, self.tac, self.fgs) = self.get_unencoded()

    def get_unencoded(self):
        data = [self.scrb.imgpix, self.scrb.specpix, self.scrb.tacpix, self.scrb.fgspix]
        return map(lambda x: pd.DataFrame.from_dict(x, orient='index'), data)

    def run_matching(self):
        for exp_type in self.exp_types:
            self.match_product_groups(exp_type)

    def match_product_groups(self, exp_type):
        for k, v in self.scrb.expdata[exp_type].items():
            exposures = list(v.keys())
            self.df.loc[self.df.dname.isin(exposures), 'pwild'] = k
            info = self.df.loc[exposures[0]]
            l3 = self.df.loc[
                (
                    self.df['params'] == info['params']
                ) & (
                    self.df['DagNodeName'].isin(self.l3_dags)
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
                pname = l3.iloc[0]['dname']
                imagesize = l3.iloc[0]['imagesize']
            self.data[exp_type].loc[k, 'pname'] = pname
            self.data[exp_type].loc[k, 'imagesize'] = imagesize
            self.df.loc[pname, 'pname'] = pname
            self.df.loc[self.df.dname.isin(exposures), 'pname'] = pname
            self.df.loc[pname, 'pwild'] = k

    def drop_mosaics(self):
        self.df['mosaic'] = self.df['dname'].apply(lambda x: self.mark_mosaics(x))
        mosaics = self.df.loc[self.df['mosaic']]
        if len(mosaics) > 0:
            mpath = f"{self.outpath}/mosaics.csv"
            if os.path.exists(mpath):
                prior = pd.read_csv(mpath, index_col=self.idxcol)
                mosaics = pd.concat([mosaics, prior], axis=0)
            mosaics.loc[:, self.idxcol] = mosaics.index
            mosaics.to_csv(mosaics, index=False)
            self.log.info(f"Mosaic data saved to: {mpath}")
            self.log.info(f"Dropping {len(mosaics.index)} mosaics from ingest data")
            self.df.drop(mosaics.index, axis=0, inplace=True)
        self.df.drop('mosaic', axis=1, inplace=True)

    def drop_unmatched_data(self):
        for exp_type in list(self.data.keys()):
            try:
                if 'imagesize' in self.data[exp_type].columns:
                    self.rem[exp_type] = self.data[exp_type].loc[self.data[exp_type]['imagesize'].isna()].copy()
                else:
                    self.rem[exp_type] = self.data[exp_type].copy()
                self.log.info(f"Dropping {len(self.rem[exp_type].index)} of {len(self.data[exp_type].index)} unmatched inputs for {exp_type}")
                self.data[exp_type].drop(self.rem[exp_type].index, axis=0, inplace=True)
            except KeyError:
                continue

    def convert_imagesize_units(self):
        for exp_type in self.exp_types:
            try:
                if 'imagesize' in self.data[exp_type].columns:
                    self.data[exp_type]['imgsize_gb'] = self.data[exp_type]['imagesize'].apply(lambda x: x / 10**6)
            except KeyError:
                continue


    def update_dags(self):
        alldags = sorted(list(self.df[self.dagcol].value_counts().index))
        self.l1_dags = [d for d in alldags if '1' in d]
        self.l3_dags = [d for d in alldags if '3' in d]

    #TODO
    def extrapolate_datasets(self, fpath=None):
        """
        #TODO
        1. load priors (unmatched L1): ingest.csv
        2. scrape new data file
        3. initial scrub on new data
        4. combine new data with priors
        5. scrub combined datasets
        6. run matching
        7. append/update training data: preprocessed.csv, train-{exp}.csv
        8. separate matched/unmatched and save/overwrite: ingest.csv, rem-{exp}.csv
        """
        fpath = self.outpath if fpath is None else fpath
        df = pd.read_csv(f"{fpath}/ingest.csv")
        df.set_index('dname', drop=False, inplace=True)
        self.df = pd.concat([self.df, df], axis=0)
        
        for exp_type in self.exp_types:
            fp_train = glob.glob(f"{fpath}/train-{exp_type.lower()}.csv")
            fp_rem = glob.glob(f"{fpath}/rem-{exp_type.lower()}.csv")
            if fp_train:
                data = pd.read_csv(fp_train[0], index_col=self.idxcol)
                if len(self.data[exp_type]) > 0:
                    self.data[exp_type] = pd.concat([self.data[exp_type], data], axis=0)
                else:
                    self.data[exp_type] = data.copy()
            if fp_rem:
                rem = pd.read_csv(fp_rem[0], index_col=self.idxcol)
                if len(self.rem[exp_type]) > 0:
                    self.rem[exp_type] = pd.concat([self.rem[exp_type], rem], axis=0)
                else:
                    self.rem[exp_type] = rem.copy()

        self.update_dags()
        params = list(self.df.loc[(self.df.pname.isna()) & (self.df.DagNodeName.isin(self.l1_dags))].params.unique())
        matched_params = {}
        for param in params:
            l3m = self.df.loc[(self.df.params == param) & (self.df.DagNodeName.isin(self.l3_dags))]
            if len(l3m) > 0:
                self.log.debug(f"MATCH: {param}")
                matched_params[param] = dict(pname=l3m.dname.values[0], imagesize=l3m.imagesize.values[0])
                # TODO
        if matched_params:
            for param, data in matched_params.items():
                pass

    def save_ingest_data(self):
        self.drop_combined_products()
        dpath = f"{self.outpath}/ingest.csv"
        self.df[self.idxcol] = self.df.index
        self.df.to_csv(dpath, index=False)
        self.log.info(f"Ingest (raw) data saved to: {dpath}")

    def save_training_sets(self):
        for exp_type in self.data.keys():
            data = self.data[exp_type].copy()
            data[self.idxcol] = data.index
            fpath = f"{self.outpath}/train-{exp_type.lower()}.csv"
            data.to_csv(fpath, index=False)
            self.log.info(f"{exp_type} training data saved to: {fpath}")
            if exp_type in list(self.rem.keys()):
                rpath = f"{self.outpath}/rem-{exp_type.lower()}.csv"
                self.rem[exp_type][self.idxcol] = self.rem[exp_type].index
                self.rem[exp_type].to_csv(rpath, index=False)
                self.log.info(f"Remaining {exp_type} data saved to: {rpath}")
        self.save_ingest_data()


if __name__ == "__main__":
    parser = ArgumentParser.parse_args(
        prog="spacekit", usage="spacekit.preprocessor.ingest input_path [options: --skope, -p, -s, -e]"
    )
    parser.add_argument("input_path", type=str, help="")
    parser.add_argument("--skope", type=str, default="jwst", help="")
    parser.add_argument("-p", "--pfx", type=str, default="", help="file name prefix to limit search on local disk")
    parser.add_argument("-s", "--sfx", type=str, default=".csv", help="file name suffix to limit search on local disk")
    parser.add_argument("-e", "--extra_data", type=str, default=None, help="")