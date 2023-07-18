"""Configuration for JWST calibration reprocessing machine learning projects.
"""
GENKEYS = [
    "PROGRAM",  # Program number
    "OBSERVTN",  # Observation number
    "BKGDTARG",  # Background target
    "VISITYPE",  # Visit type
    "TSOVISIT",  # Time Series Observation visit indicator
    "TARGNAME",  # Standard astronomical catalog name for target
    "TARG_RA",  # Target RA at mid time of exposure
    "TARG_DEC",  # Target Dec at mid time of exposure
    "INSTRUME",  # Instrument used to acquire the data
    "DETECTOR",  # Name of detector used to acquire the data
    "FILTER",  # Name of the filter element used
    "PUPIL",  # Name of the pupil element used
    "EXP_TYPE",  # Type of data in the exposure
    "CHANNEL",  # Instrument channel
    "SUBARRAY",  # Subarray used
    "NUMDTHPT",  # Total number of points in pattern
    "GS_RA",  # guide star right ascension
    "GS_DEC",  # guide star declination
]

SCIKEYS = [
    "RA_REF",
    "DEC_REF",
    "CRVAL1",
    "CRVAL2",
]

COLUMN_ORDER = {
    "asn": [
        "instr",
        "detector",
        "exp_type",
        "visitype",
        "filter",
        "pupil",
        "channel",
        "subarray",
        "bkgdtarg",
        "tsovisit",
        "nexposur",
        "numdthpt",
        "offset",
        "max_offset",
        "mean_offset",
        "sigma_offset",
        "err_offset",
        "sigma1_mean",
        "frac",
    ]
}

NORM_COLS = {
    "asn": [
        "offset",
        "max_offset",
        "mean_offset",
        "sigma_offset",
        "err_offset",
        "sigma1_mean",
        "frac",
    ],
}

RENAME_COLS = {
    "asn": [],
}

X_NORM = {"asn": []}

KEYPAIR_DATA = {
    "instr": {"FGS": 0, "MIRI": 1, "NIRCAM": 2, "NIRISS": 3, "NIRSPEC": 4},
    "detector": {
        "NONE": 0,
        "GUIDER1": 1,
        "GUIDER1|GUIDER2": 2,
        "GUIDER2": 3,
        "MIRIFULONG": 4,
        "MIRIFULONG|MIRIFUSHORT": 5,
        "MIRIFULONG|MIRIFUSHORT|MIRIMAGE": 6,
        "MIRIFUSHORT": 7,
        "MIRIMAGE": 8,
        "NIS": 9,
        "NRCA1": 10,
        "NRCA1|NRCA2|NRCA3|NRCA4": 11,
        "NRCA1|NRCA2|NRCA3|NRCA4|NRCB1|NRCB2|NRCB3|NRCB4": 12,
        "NRCA1|NRCA3": 13,
        "NRCA1|NRCA3|NRCALONG": 14,
        "NRCA2": 15,
        "NRCA3": 16,
        "NRCA4": 17,
        "NRCALONG": 18,
        "NRCALONG|NRCBLONG": 19,
        "NRCB1": 20,
        "NRCB1|NRCB2|NRCB3|NRCB4": 21,
        "NRCB1|NRCBLONG": 22,
        "NRCB2": 23,
        "NRCB2|NRCB4": 24,
        "NRCB3": 25,
        "NRCB4": 26,
        "NRCBLONG": 27,
        "NRS1": 28,
        "NRS1|NRS2": 29,
        "NRS2": 30,
    },
    "filter": {
        "NONE": 0,
        "CLEAR": 1,
        "F070LP": 2,
        "F070W": 3,
        "F090W": 4,
        "F1000W": 5,
        "F100LP": 6,
        "F1065C": 7,
        "F110W": 8,
        "F1130W": 9,
        "F1140C": 10,
        "F115W": 11,
        "F1280W": 12,
        "F140M": 13,
        "F140X": 14,
        "F1500W": 15,
        "F150W": 16,
        "F150W2": 17,
        "F1550C": 18,
        "F170LP": 19,
        "F1800W": 20,
        "F182M": 21,
        "F187N": 22,
        "F200W": 23,
        "F2100W": 24,
        "F210M": 25,
        "F212N": 26,
        "F2300C": 27,
        "F250M": 28,
        "F2550W": 29,
        "F2550WR": 30,
        "F277W": 31,
        "F290LP": 32,
        "F300M": 33,
        "F322W2": 34,
        "F335M": 35,
        "F356W": 36,
        "F360M": 37,
        "F380M": 38,
        "F410M": 39,
        "F430M": 40,
        "F444W": 41,
        "F460M": 42,
        "F480M": 43,
        "F560W": 44,
        "F770W": 45,
        "FND": 46,
        "GR150C": 47,
        "GR150R": 48,
        "OPAQUE": 49,
        "P750L": 50,
        "WLP4": 51,
    },
    "pupil": {
        "NONE": 0,
        "CLEAR": 1,
        "CLEARP": 2,
        "F090W": 3,
        "F115W": 4,
        "F140M": 5,
        "F150W": 6,
        "F158M": 7,
        "F162M": 8,
        "F164N": 9,
        "F200W": 10,
        "F323N": 11,
        "F405N": 12,
        "F466N": 13,
        "F470N": 14,
        "FLAT": 15,
        "GDHS0": 16,
        "GDHS60": 17,
        "GR700XD": 18,
        "GRISMC": 19,
        "GRISMR": 20,
        "MASKBAR": 21,
        "MASKIPR": 22,
        "MASKRND": 23,
        "NRM": 24,
        "WLM8": 25,
        "WLP8": 26,
    },
    "exp_type": {
        "NONE": 0,
        "FGS_FOCUS": 1,
        "FGS_IMAGE": 2,
        "FGS_INTFLAT": 3,
        "MIR_4QPM": 4,
        "MIR_FLATIMAGE": 5,
        "MIR_FLATIMAGE-EXT": 6,
        "MIR_FLATMRS": 7,
        "MIR_IMAGE": 8,
        "MIR_LRS-FIXEDSLIT": 9,
        "MIR_LRS-FIXEDSLIT|NIS_SOSS": 10,
        "MIR_LRS-SLITLESS": 11,
        "MIR_LYOT": 12,
        "MIR_MRS": 13,
        "NIS_AMI": 14,
        "NIS_DARK": 15,
        "NIS_EXTCAL": 16,
        "NIS_IMAGE": 17,
        "NIS_LAMP": 18,
        "NIS_SOSS": 19,
        "NIS_WFSS": 20,
        "NRC_CORON": 21,
        "NRC_DARK": 22,
        "NRC_GRISM": 23,
        "NRC_IMAGE": 24,
        "NRC_LED": 25,
        "NRC_TSGRISM": 26,
        "NRC_TSIMAGE": 27,
        "NRC_WFSS": 28,
        "NRC_WFSS|NRS_AUTOFLAT|NRS_AUTOWAVE": 29,
        "NRS_AUTOFLAT": 30,
        "NRS_AUTOFLAT|NRS_AUTOWAVE|NRS_FIXEDSLIT": 31,
        "NRS_AUTOWAVE": 32,
        "NRS_AUTOWAVE|NRS_IFU": 33,
        "NRS_BRIGHTOBJ": 34,
        "NRS_FIXEDSLIT": 35,
        "NRS_IFU": 36,
        "NRS_LAMP": 37,
        "NRS_MIMF": 38,
        "NRS_MSASPEC": 39,
    },
    "channel": {"NONE": 0, "12": 1, "34": 2, "LONG": 3, "SHORT": 4},
    "subarray": {
        "NONE": 0,
        "ALLSLITS": 1,
        "BRIGHTSKY": 2,
        "FULL": 3,
        "MASK1065": 4,
        "MASK1140": 5,
        "MASK1550": 6,
        "MASKLYOT": 7,
        "SLITLESSPRISM": 8,
        "SUB128": 9,
        "SUB160": 10,
        "SUB160P": 11,
        "SUB2048": 12,
        "SUB256": 13,
        "SUB32": 14,
        "SUB320": 15,
        "SUB320A335R": 16,
        "SUB320A430R": 17,
        "SUB320ALWB": 18,
        "SUB32TATS": 19,
        "SUB32TATSGRISM": 20,
        "SUB400P": 21,
        "SUB512": 22,
        "SUB512S": 23,
        "SUB64": 24,
        "SUB640": 25,
        "SUB640A210R": 26,
        "SUB640ASWB": 27,
        "SUB64FP1A": 28,
        "SUB64P": 29,
        "SUB80": 30,
        "SUB96DHSPILA": 31,
        "SUBAMPCAL": 32,
        "SUBFSA210R": 33,
        "SUBFSA335R": 34,
        "SUBFSA430R": 35,
        "SUBFSALWB": 36,
        "SUBFSASWB": 37,
        "SUBGRISM128": 38,
        "SUBGRISM256": 39,
        "SUBGRISM64": 40,
        "SUBNDA210R": 41,
        "SUBNDA335R": 42,
        "SUBNDA430R": 43,
        "SUBNDALWBL": 44,
        "SUBNDALWBS": 45,
        "SUBNDASWBS": 46,
        "SUBS200A1": 47,
        "SUBS200A2": 48,
        "SUBS400A1": 49,
        "SUBSTRIP256": 50,
        "SUBSTRIP96": 51,
        "SUBTAAMI": 52,
        "SUBTASOSS": 53,
        "WFSS128C": 54,
        "WFSS128R": 55,
        "WFSS64C": 56,
        "WFSS64R": 57,
    },
    "visitype": {
        "NONE": 0,
        ".+WFSC.+": 1,
        "PARALLEL_PURE": 2,
        "PARALLEL_SLEW_CALIBRATION": 3,
        "PRIME_TARGETED_FIXED": 4,
        "PRIME_TARGETED_MOVING": 5,
        "PRIME_UNTARGETED": 6,
        "PRIME_WFSC_ROUTINE": 7,
        "PRIME_WFSC_SENSING_CONTROL": 8,
        "PRIME_WFSC_SENSING_ONLY": 9,
    },
}
