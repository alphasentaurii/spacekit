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
    def __init__(self, input_path=None, pfx="", sfx=".csv", outpath=None, name="JwstCalIngest", **log_kws):
        self.input_path = input_path
        self.pfx = pfx
        self.sfx = sfx
        self.outpath = input_path if outpath is None else outpath
        self.exp_types = ["IMAGE", "SPEC", "TAC"]
        self.files = None
        self.index = "Dataset"
        self.dagcol = "DagNodeName"
        self.df = None
        self.l1_dags = []
        self.l3_dags = []
        self.input_dict = None
        self.input_data = {}
        self.product_matches = None
        self.unmatched = {}
        # *** TEMP *** #
        self.asn_mode = False
        self.scrubber = None
        self.__name__ = name
        self.log = Logger(self.__name__, **log_kws).spacekit_logger()
    
    def run_ingest(self, file_suffix="-2024"):
        self.read_files()
        if len(self.files) == 0:
            return
        self.ingest_data()
        self.initial_scrub()
        self.scrub_exposures()
        self.match_l1_to_l3()
        self.drop_incomplete_data()
        self.convert_imagesize_units()
        self.save_training_sets(
            self,
            sfx=file_suffix,
            unmatched=True,
            raw=True,
            drop_c=True,
            save=True
        )
    
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
        l1_dags = [d for d in alldags if '1' in d]
        l3_dags = [d for d in alldags if '3' in d]
        dags_l1_l3 = l1_dags + l3_dags
        df = df.loc[df[self.dagcol].isin(dags_l1_l3)]
        self.l1_dags.extend([l for l in l1_dags if l not in self.l1_dags])
        self.l3_dags.extend([l for l in l3_dags if l not in self.l3_dags])
        return df

    def ingest_data(self):
        for i, f in enumerate(self.files):
            df = pd.read_csv(f, index_col=self.index)
            df = self.drop_l2_data(df)
            filedate = os.path.basename(f).split('_')[0]
            df['date'] = filedate
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

    def validate_obs(self, x):
        if not isinstance(x, str):
            x = str(x)
        if len(x) < 3:
            if len(x) == 1:
                x = f"00{x}"
            else:
                x = f"0{x}"
        return x

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
        # *** TEMP *** #
        if self.asn_mode is True:
            self.df['OBSERVTN'] = self.df['dname'].apply(lambda x: self.extract_obs(x))
        # *** end TEMP *** #
        self.df['OBSERVTN'] = self.df['OBSERVTN'].apply(lambda x: self.validate_obs(x))
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

    def scrub_exposures(self):
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
    
    def match_l1_to_l3(self):
        # *** TEMP *** #
        if self.asn_mode is True:
            self.match_asn_products()
        # *** end TEMP *** #
        else:
            for exp_type, products in list(zip(["IMAGE", "SPEC", "TAC"], [
                self.scrubber.img_products,
                self.scrubber.spec_products,
                self.scrubber.tac_products
                ])):
                self.match_product_groups(products, exp_type=exp_type)


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
                pname = l3.iloc[0]['dname']
                imagesize = l3.iloc[0]['imagesize']
            self.input_data[exp_type].loc[k, 'pname'] = pname
            self.input_data[exp_type].loc[k, 'imagesize'] = imagesize
            self.df.loc[pname, 'pname'] = pname
            for e in list(v.keys()):
                self.df.loc[e, 'pname'] = pname

    def match_asn_products(self):
        def asn_wildcard(x):
            asndate = x.split('_')[1]
            asnwild = x.replace(asndate, '*')
            return asnwild
        self.df.loc[~self.df.DagNodeName.isin(self.l1_dags)]
        radio = JwstCalRadio()
        self.product_matches = radio.match_asn_filename(self.input_data)
        for exp_type, products in list(zip(["IMAGE", "SPEC", "TAC"], [
            self.scrubber.img_products,
            self.scrubber.spec_products,
            self.scrubber.tac_products
            ])):
            for k, v in products.items():
                info = self.product_matches[exp_type].get(k, None)
                if info:
                    try:
                        imagesize = self.df.loc[self.df.dname == info['asn']]['imagesize'].values[0]
                    except IndexError:
                        imagesize = None # no L3 product
                    for key, value in info.items():
                        self.input_data[exp_type].loc[k, key.lower()] = value
                        for e in list(v.keys()):
                            self.df.loc[e, key] = value

                    if imagesize is not None:
                        self.df.loc[info['asn'], 'pname'] = info['pname']
                        self.df.loc[info['asn'], 'TARGNAME'] = info['TARGNAME']
                        self.input_data[exp_type].loc[k, 'imagesize'] = imagesize

    def drop_combined_products(self, save=True):
        l3 = self.df.loc[~self.df.DagNodeName.isin(self.l1_dags)].dname.values
        c = [p for p in l3 if p.split('-')[1][0] == 'c']
        if len(c) > 0:
            cdata = self.df.loc[self.df.dname.isin(c)]
            if save is True:
                cdata.loc[:, self.index] = cdata.index
                cdata.to_csv(f"{self.outpath}/c1000.csv", index=False)
                self.log.info(f"c1000 data saved to: ")
            self.log.info(f"Dropping {len(c)} c1000 products")
            self.df.drop(c, axis=0, inplace=True)

    def drop_incomplete_data(self):
        for exp_type in list(self.input_data.keys()):
            try:
                if 'imagesize' in self.input_data[exp_type].columns:
                    self.unmatched[exp_type] = self.input_data[exp_type].loc[self.input_data[exp_type]['imagesize'].isna()]
                else:
                    self.unmatched[exp_type] = self.input_data[exp_type]
                self.log.info(f"Dropping {len(self.unmatched[exp_type].index)} of {len(self.input_data[exp_type].index)} incomplete inputs for {exp_type}")
                self.input_data[exp_type].drop(self.unmatched[exp_type].index, axis=0, inplace=True)
            except KeyError:
                continue

    def convert_imagesize_units(self):
        for exp_type in self.exp_types:
            try:
                if 'imagesize' in self.input_data[exp_type].columns:
                    self.input_data[exp_type]['imgsize_gb'] = self.input_data[exp_type]['imagesize'].apply(lambda x: x / 10**6)
            except KeyError:
                continue

    def save_raw_dataset(self, sfx="", drop_c=True, save=True):
        if drop_c is True:
            self.drop_combined_products(save=save)
        self.df[:, self.index] = self.df.index
        self.df.to_csv(f"{self.outpath}/ingest{sfx}.csv", index=False)

    def save_training_sets(self, sfx="-2024", unmatched=True, raw=True, drop_c=True, save=True):
        for exp_type in list(self.input_data.keys()):
            data = self.input_data[exp_type].copy()
            data.loc[:, self.index] = data.index
            data.to_csv(f"{self.outpath}/train-{exp_type.lower()}{sfx}.csv", index=False)
            if unmatched is True and exp_type in list(self.unmatched.keys()):
                self.unmatched[exp_type].to_csv(f"{self.outpath}/unmatched-{exp_type.lower()}{sfx}.csv", index=False)
        if raw is True:
            self.save_raw_dataset(sfx=sfx, drop_c=drop_c, save=save)




if __name__ == "__main__":
    parser = ArgumentParser.parse_args(
        prog="spacekit", usage="spacekit.preprocessor.ingest input_path [options: --skope, -p, -s, -e]"
    )
    parser.add_argument("input_path", type=str, help="")
    parser.add_argument("--skope", type=str, default="jwst", help="")
    parser.add_argument("-p", "--pfx", type=str, default="", help="file name prefix to limit search on local disk")
    parser.add_argument("-s", "--sfx", type=str, default=".csv", help="file name suffix to limit search on local disk")
    parser.add_argument("-e", "--extra_data", type=str, default=None, help="")