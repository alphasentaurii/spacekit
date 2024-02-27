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
    def __init__(self, input_path=None, pfx="", outpath=None, **log_kws):
        """Loads raw JWST Calibration Pipeline metadata from local disk (`input_path`)
        and runs initial ML preprocessing steps necessary prior to model training. The resulting 
        dataframes will be "ingested" into any pre-existing training sets located in `outpath`. 
        This outpath acts as the primary database containing several "tables" (dataframes stored
        in .csv files). This class is designed to run on single or multiple files at a time 
        (limit specificity using 'pfx`). 

        Input file naming convention: YYYY-MM-DD_%d.csv (%d = day of year) ex: 2024-02-21_052.csv
        Alternate formats currently not supported because filenames are used to store date info.
        Examples: To ingest multiple files from November 2023, set `pfx="2023-11". To ingest only
        one file from January 3, 2024, set `pfx="2024-01-03". You can also pass in a wildcard, for
        example `pfx="*_3" would search for all data collected on days 300-365 of any year,
        while `pfx="2023*_3" would do the same but only for the year 2023.

        The contents of raw metadata files are expected to contain:
 
            1) columns consistent with Fits header keyword-values used in JWST Cal model training 
            (see `spacekit.skopes.jwst.cal.config`) 

            2) rows of Level 1/1b exposures (inputs/features) along with Level 3 products

            3) imagesize (memory footprint) for each L3 product (outputs/target)
        
        At STSCI, additional model training data is acquired daily from the telescope's calibration pipeline.
        Due to the nature of an automated 24-hour data collection cycle, some Level 3 products may still be
        processing at the time data is collected. This results in a given input file containing groups of L1
        exposures with no matching L3 product. JwstCalIngest will run preprocessing on all L1 inputs and attempt 
        to match them with an L3 product in the same file. Any complete datasets (where a match is identified) are
        inserted into the "database", a file called `training.csv`. Any remaining L1 exposures that did not 
        find a match are stored into a separate "table" called `ingest.csv`. The next time this ingest process
        is run, the script will load both the new data as well as prior (unmatched) data. The assumption here is
        that the missing L3 product(s) (and sometimes even additional L1 exposures for this association) will 
        eventually complete the pipeline and show up in subsequent files.

        Additional output files are model-specific encoded subsets of `preprocessed` and `ingest`. Data is inserted
        into these in the same manner as appropriate. The actual files to be used for model training are named as
        "train-{modelname}.csv", while `training.csv` contains all the original columns with unencoded values
        and is intended to be used primarily for data analysis and debugging purposes.

        Database: {outpath}

        Tables: {.csv files}

            Accumulated data storing unencoded values
            - preprocessed:  complete L1-L3 groupings
            - ingest: unmatched L1 exposures
            - mosaics: c1XXX association candidate L3 products (currently not supported)

            Encoded datasets finalized and ready for model training (input features + y-targets)
            - train-image: L3 image model
            - train-spec: L3 spectroscopy model
            - train-tac: L3 TSO/AMI/CORON model

            Encoded input features of remaining L1 exposures (y-targets pending)
            - rem-image.csv
            - rem-spec.csv
            - rem-tac.csv

        Parameters
        ----------
        input_path : str (path), optional
            directory path to csv files on local disk, by default None (current working directory)
        pfx : str, optional
            filename start pattern (e.g. "2023" or "*-12-), by default ""
        outpath : str (path), optional
            directory path to save (and/or update) preprocessed files on local disk, by default None (current working directory)
        """
        self.input_path = input_path.rstrip("/")
        self.pfx = pfx
        self.outpath = input_path if outpath is None else outpath.rstrip("/")
        self.exp_types = ["IMAGE", "SPEC", "TAC"]
        self.files = []
        self.idxcol = "Dataset"
        self.dag = "DagNodeName"
        self.df = None
        self.l1_dags = []
        self.l3_dags = []
        self.data = {}
        self.product_matches = None
        self.exmatches = {}
        self.rem = {}
        self.param_cols = ['pid', 'OBSERVTN', 'FILTER', 'GRATING', 'PUPIL', 'SUBARRAY', 'EXP_TYPE']
        self.scrb = None
        self.trainpath = self.outpath + "/train-{}.csv"
        self.rempath =  self.outpath + "/rem-{}.csv"
        self.__name__ = "JwstCalIngest"
        self.log = Logger(self.__name__, **log_kws).spacekit_logger()

    @property
    def float_cols(self):
        return self._float_cols()

    def _float_cols(self):
        return [
            'CRVAL1',
            'CRVAL2',
            'RA_REF',
            'DEC_REF',
            'GS_RA',
            'GS_DEC',
            'TARG_RA',
            'TARG_DEC'
        ]

    def run_ingest(self, apriori=True, save_l1=True):
        self.ingest_data()
        if len(self.files) == 0:
            return
        self.initial_scrub()
        if apriori is True:
            self.load_priors()
        self.scrub_exposures()
        self.extrapolate()
        self.save_ingest_data(save_l1=save_l1)
        self.save_training_sets()

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
        alldags = sorted(list(df[self.dag].value_counts().index))
        l1_dags = [d for d in alldags if '1' in d]
        l3_dags = [d for d in alldags if '3' in d]
        dags_l1_l3 = l1_dags + l3_dags
        df = df.loc[df[self.dag].isin(dags_l1_l3)]
        self.l1_dags.extend([l for l in l1_dags if l not in self.l1_dags])
        self.l3_dags.extend([l for l in l3_dags if l not in self.l3_dags])
        return df


    def load_and_recast(self, dpath, idxcol=None):
        if not os.path.exists(dpath):
            self.log.warning(f"File does not exist at specified path: {dpath}")
            return
        idxcol = self.idxcol if idxcol is None else idxcol
        df = pd.read_csv(dpath, index_col=idxcol)
        return self.recast_dtypes(df)

    def recast_dtypes(self, df):
        """When loading a saved dataframe, some datatypes need to be recast appropriately
        in order to be able to edit existing / insert new values."""
        df['OBSERVTN'] = df['OBSERVTN'].apply(lambda x: self.validate_obs(x))
        df['PROGRAM'] = df['PROGRAM'].apply(lambda x: '{:0>5}'.format(x))
        for col in self.float_cols:
            df[col] = df[col].apply(lambda x: self.convert_to_float(x))
        df['year'] = df['year'].astype('int64')
        df['date'] = pd.to_datetime(df['date'], yearfirst=True)
        return df

    def load_priors(self, fname="ingest.csv"):
        ingest_file = os.path.join(self.outpath, fname)
        if not os.path.exists(ingest_file):
            self.log.debug("Prior data not found -- skipping.")
            return None
        self.log.info("Collecting prior data")
        di = self.load_and_recast(ingest_file)
        try:
            self.df = pd.concat([self.df, di], axis=0)
            self.log.info(f"Prior data loaded successfully: {len(di)} exposures added.")
            self.update_dags()
            self.df.drop_duplicates(subset='dname', keep='first', inplace=True)
        except Exception as e:
            self.log.error(str(e))

    def update_dags(self):
        alldags = sorted(list(self.df[self.dag].value_counts().index))
        self.l1_dags = [d for d in alldags if '1' in d]
        self.l3_dags = [d for d in alldags if '3' in d]

    def initial_scrub(self):
        if self.df is None:
            return
        self.log.info(f"{len(self.df)} datasets loaded")
        self.df['dname'] = self.df.index
        self.df['dname'] = self.df['dname'].apply(lambda x: self.strip_file_suffix(x))
        self.df.rename({'ImageSize':'imagesize', self.dag: 'dag'}, axis=1, inplace=True)
        self.dag = 'dag'
        self.drop_duplicates()
        self.df.set_index('dname', drop=False, inplace=True)
        self.df['pid'] = self.df['dname'].apply(lambda x: self.extract_pid(x))
        self.df = self.recast_dtypes(self.df)
        params = list(map(lambda x: '-'.join([str(y) for y in x if y != "NONE"]),  self.df[self.param_cols].values))
        self.df['params'] = pd.DataFrame(params, index=self.df.index)
        self.drop_mosaics()
        self.df.drop('Dataset', axis=1, inplace=True)

    def drop_duplicates(self, priority='imagesize'):
        if priority is None:
            self.df.drop_duplicates(subset='dname', inplace=True)
            return
        dupes = sorted(list(self.df.loc[self.df.duplicated(subset='dname')].index))
        self.df.loc[dupes, 'dupe'] = True
        self.df.loc[self.df.dupe.isna(), 'dupe'] = False
        for d in dupes:
            max_priority = self.df.loc[d][priority].max()
            self.df.loc[(self.df.dname == d) & (self.df[priority] == max_priority), 'dupe'] = False
        self.df.reset_index(inplace=True)
        self.log.info(f"Dropping {len(dupes)} duplicates (priority={priority})")
        self.df.drop(self.df.loc[self.df.dupe].index, axis=0, inplace=True)
        self.df.drop('dupe', axis=1, inplace=True)
        if len(sorted(list(self.df.loc[self.df.duplicated(subset='dname')].index))) > 0:
            self.df.drop_duplicates(subset='dname', inplace=True)

    @staticmethod
    def strip_file_suffix(x):
        if x.endswith("fits"):
            x = '_'.join(x.split('_')[:-1])
        return x

    @staticmethod
    def extract_pid(x):
        if not isinstance(x, str):
            return x
        pid = x[2:7]
        if pid[0] == '0':
            pid = pid[1:]
        return int(pid)

    @staticmethod
    def validate_obs(x):
        return '{:0>3}'.format(x)

    @staticmethod
    def convert_to_float(x):
        if x != "NONE":
            return float(x)

    @staticmethod
    def mark_mosaics(x):
        if len(x.split('-')) < 2:
            return False
        elif x.split('-')[1][0] != 'c':
            return False
        return True

    def drop_mosaics(self):
        self.df['mosaic'] = self.df['dname'].apply(lambda x: self.mark_mosaics(x))
        mosaics = self.df.loc[self.df['mosaic']].copy()
        if len(mosaics) > 0:
            mpath = f"{self.outpath}/mosaics.csv"
            kwargs = dict(mode='a', index=False, header=False) if os.path.exists(mpath) else dict(index=False)
            mosaics[self.idxcol] = mosaics.index
            mosaics.to_csv(mpath, **kwargs)
            self.log.info(f"Mosaic data saved to: {mpath}")
            self.log.info(f"Dropping {len(mosaics.index)} mosaics from ingest data")
            self.df.drop(mosaics.index, axis=0, inplace=True)
        self.df.drop('mosaic', axis=1, inplace=True)

    def scrub_exposures(self):
        self.scrb = JwstCalScrubber(
                self.input_path,
                data=self.df.loc[self.df[self.dag].isin(self.l1_dags)],
                encoding_pairs=KEYPAIR_DATA,
                mode='df'
            )
        nonsci = self.df.loc[~self.df['EXP_TYPE'].isin(self.scrb.level3_types)]
        self.df.drop(nonsci.index, axis=0, inplace=True)
        self.log.info(f"Dropped {len(nonsci)} non-L3 exposure types")
        for exp_type in self.exp_types:
            inputs = self.scrb.scrub_inputs(exp_type=exp_type)
            if inputs is not None:
                inputs['dname'] = inputs.index
                self.data[exp_type] = inputs
        (self.img, self.spec, self.tac, self.fgs) = self.get_unencoded()

    def get_unencoded(self):
        data = [self.scrb.imgpix, self.scrb.specpix, self.scrb.tacpix, self.scrb.fgspix]
        return map(lambda x: pd.DataFrame.from_dict(x, orient='index'), data)

    def extrapolate(self):
        for exp_type in self.exp_types:
            self.match_product_groups(exp_type)
        self.drop_extras("SPEC")
        self.drop_unmatched()
        self.convert_imagesize_units()
        if 'pname' not in self.df.columns:
            self.log.debug("No L3 candidates to match")
            return
        self.update_repro()

    def match_product_groups(self, exp_type):
        self.exmatches[exp_type] = {}
        for k, v in self.scrb.expdata[exp_type].items():
            exposures = list(v.keys())
            info = self.df.loc[exposures[0]]
            l3 = self.df.loc[
                (
                    self.df['params'] == info['params']
                ) & (
                    self.df[self.dag].isin(self.l3_dags)
                )
            ]
            if len(l3) == 0:
                self.log.debug(f"No matching products identified: {k}")
                self.df.loc[self.df.dname.isin(exposures), 'expmode'] = exp_type
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
                self.exmatches[exp_type][info['params']] = [p for p in pnames if p != pname]
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
            self.df.loc[self.df.pname == pname, 'expmode'] = exp_type

    def drop_extras(self, exp_type):
        """Drop other L3 channel products for MIR_MRS (only 1 of 4 to be used in model training)"""
        drops = []
        for _, pnames in self.exmatches[exp_type].items():
            if len(pnames) == 3:
                drops.extend(pnames)
        self.log.info(f"Dropping {len(drops)} extra products for {len(drops)/3} MIR_MRS matches")
        self.df.drop(drops, axis=0, inplace=True)

    def drop_unmatched(self):
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

    def convert_imagesize_units(self, data=None):
        if data is not None:
            data['imgsize_gb'] = data['imagesize'].apply(lambda x: x / 10**6)
            return data
        for exp_type in self.exp_types:
            try:
                if 'imagesize' in self.data[exp_type].columns:
                    self.data[exp_type]['imgsize_gb'] = self.data[exp_type]['imagesize'].apply(lambda x: x / 10**6)
            except KeyError:
                continue

    def update_repro(self):
        l3 = self.df.loc[(self.df.pname.isna()) & (self.df.dag.isin(self.l3_dags))]
        if len(l3) == 0:
            self.log.debug("No repro candidates found - skipping.")
            return
        self.log.info(f"Identified {len(l3)} potential reprocessed products eligible for update")
        dp = self.load_and_recast(f"{self.outpath}/training.csv")
        if dp is None:
            self.log.warning("Could not update repro data - file not found.")
            return
        updates = {}
        pnames = list(l3.index)
        for pname in pnames:
            try:
                expmode = dp.loc[pname]['expmode']
                if expmode not in updates:
                    updates[expmode] = [pname]
                else:
                    updates[expmode].append(pname)
            except KeyError:
                continue
        for exp_type, pnames in updates.items():
            repro = dict()
            for pname in pnames:
                repro[pname] = dict(
                    imagesize=l3.loc[pname].imagesize,
                    doy=l3.loc[pname].doy,
                    date=l3.loc[pname].date,
                    year=l3.loc[pname].year
                )
            data = pd.read_csv(self.trainpath.format(exp_type.lower()), index_col=self.idxcol)
            for pname, revised in repro.items():
                for k, v in revised.items():
                    dp.loc[pname, k] = v
                data.loc[data.pname == pname, 'imagesize'] = revised['imagesize']
            data = self.convert_imagesize_units(data=data)
            data[self.idxcol] = data.index
            data.to_csv(self.trainpath.format(exp_type.lower()), index=False)
            self.log.info(f"Updated {len(repro)} reprocessed {exp_type} products.")
        dp[self.idxcol] = dp.index
        dp.to_csv(f"{self.outpath}/training.csv", index=False)
        self.df.drop(l3.index, axis=0, inplace=True)
        self.log.info(f"Preprocessed file updated and L3 repro products removed from dataframe.")

    def save_training_sets(self):
        for exp_type in self.exp_types:
            if exp_type in self.data.keys() and len(self.data[exp_type]) > 0:
                fpath = self.trainpath.format(exp_type.lower())
                self.data[exp_type][self.idxcol] = self.data[exp_type].index
                kwargs = dict(mode='a', index=False, header=False) if os.path.exists(fpath) else dict(index=False)
                self.data[exp_type].to_csv(fpath, **kwargs)
                self.log.info(f"{exp_type} training data saved to: {fpath}")
            if exp_type in self.rem.keys() and len(self.rem[exp_type]) > 0:
                rpath = self.rempath.format(exp_type.lower())
                self.rem[exp_type][self.idxcol] = self.rem[exp_type].index
                self.rem[exp_type].to_csv(rpath, index=False)
                self.log.info(f"Remaining {exp_type} data saved to: {rpath}")

    def save_ingest_data(self, save_l1=True):
        self.df[self.idxcol] = self.df.index
        ingest_file = f"{self.outpath}/ingest.csv"
        if 'pname' not in self.df.columns:
            di = self.df.loc[self.df.dag.isin(self.l1_dags)]
        else:
            di = self.df.loc[self.df.pname.isna()].copy()
            di.drop(['pname'], axis=1, inplace=True)
        
        di.to_csv(ingest_file, index=False)
        self.log.info(f"Remaining Ingest data saved to: {ingest_file}")

        dp = self.df.drop(di.index, axis=0)
        if len(dp) > 0:
            if save_l1 is True:
                l1 = dp.loc[dp.dag.isin(self.l1_dags)]
                l1_path = f"{self.outpath}/level1.csv"
                kwargs = dict(mode='a', index=False, header=False) if os.path.exists(l1_path) else dict(index=False)
                l1.to_csv(l1_path, **kwargs)

            dp = dp.loc[dp.dag.isin(self.l3_dags)]
            ppath = f"{self.outpath}/training.csv"
            kwargs = dict(mode='a', index=False, header=False) if os.path.exists(ppath) else dict(index=False)
            dp.to_csv(ppath, **kwargs)
            self.log.info(f"{len(dp)} L3 products added to: {ppath}")


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="spacekit", usage="spacekit.preprocessor.ingest input_path [options: --skope, -p, -o]"
    )
    parser.add_argument("input_path", type=str, help="")
    parser.add_argument("--skope", type=str, default="jwst", help="")
    parser.add_argument("--pfx", "-p", type=str, default="", help="file name prefix to limit search on local disk")
    parser.add_argument("--outpath", "-o", type=str, default=None, help="path to save preprocessed ingest files on local disk")
    parser.add_argument("--apriori", "-a", action="store_true", help="include prior unmatched L1 data from outpath")
    parser.add_argument("--level1", "-l", action="store_true", help="save matched level 1 input data to separate file")
    args = parser.parse_args()
    if args.skope.lower() == "jwst":
        os.makedirs(args.outpath, exist_ok=True)
        kwargs = dict(input_path=args.input_path, pfx=args.pfx, outpath=args.outpath)
        jc = JwstCalIngest(**kwargs)
        jc.run_ingest(apriori=args.apriori, save_l1=args.level1)
