import pandas as pd
import psycopg2
import numpy as np
from astropy.coordinates import SkyCoord, match_coordinates_sky
import multiprocessing as mp
from astropy import units as u
import sys
from tqdm import tqdm

def process_entry(args):
    idx, ra, dec, data_release, radius = args

    # connect to database
    db = psycopg2.connect(host='decatdb.lbl.gov', database='desidb', user='desi', password = "5kFibers!", port="5432")
    cursor = db.cursor()

    observed = "N"
    if np.isnan(ra) or np.isnan(dec):
        match = np.empty((11,))
        match[:] = np.nan
    else:
        query = f'SELECT f.target_ra, f.target_dec, r.z, r.zerr, r.zwarn, r.spectype, f.tileid, f.petal_loc, f.targetid, f.fiber, c.night \n' \
                f'FROM {data_release}.tiles_fibermap f\n' \
                f'INNER JOIN {data_release}.cumulative_tiles c ON f.cumultile_id=c.id\n' \
                f'INNER JOIN {data_release}.tiles_redshifts r ON r.cumultile_id=c.id AND r.targetid=f.targetid\n' \
                f'WHERE q3c_radial_query(f.target_ra, f.target_dec, {ra}, {dec}, {radius});'
        
        cursor.execute(query)
        targets = cursor.fetchall()

        if len(targets) >= 1:
            targets = np.array(targets)
            zwarn = '1'

            while zwarn != '0' and len(targets) > 0:
                ra_lst, dec_lst = targets[:, 0].astype(np.float64).tolist(), targets[:, 1].astype(np.float64).tolist()
                match_idx = match_coordinates_sky(SkyCoord(ra * u.deg, dec * u.deg),
                                                  SkyCoord(ra_lst * u.deg, dec_lst * u.deg),
                                                  nthneighbor=1)[0]
                match = targets[match_idx]
                zwarn = match[4]
                if zwarn != '0':
                    targets = np.delete(targets, match_idx, 0)
        
        if len(targets) == 0:
            match = np.empty((11,))
            match[:] = np.nan
        else:
            observed = "Y"

    cursor.close()
    db.close()
    
    return idx, match[2], match[3], match[5], observed, match[6], match[7], match[8], match[9], match[10]

def main():
    if (len(sys.argv) != 6):
        print("crossmatch_to_desi_db.py - Crossmatch a catalog with observations from DESI.")
        print("Usage: python crossmatch_to_desi_db.py [catalag .fits filename with columns 'RA' and 'DEC' in degrees] [DESI data reduction] [search radius in arcsec] [output file]]")
        print("Example: crossmatch_to_desi_db.py /my/path/my_transients.fits daily 1.6 /my/path/output.fits 16")
        sys.exit(0)

    filename = sys.argv[1]
    data_release = sys.argv[2]
    radius = float(sys.argv[3])
    outfile = sys.argv[4]
    cores = int(sys.argv[5])

    # Read in catalog
    catalog = pd.read_csv(filename, low_memory=False) #skiprows = 1, 

    # Query parameters
    radius = radius / 3600.0 # arcseconds -> degrees

    obs_status = []  
    z_lst = []
    zerr_lst = []
    spectype_lst = []
    tileid_lst = []
    petal_loc_lst = []
    targetid_lst = []
    fiber_lst = []
    night_lst = []

    #print(catalog.columns)
    #for idx, (ra, dec) in enumerate(zip(catalog["ra"], catalog["declination"])):
    #    print(ra, dec)
        
    with mp.Pool(processes=cores) as pool:  # Adjust the number of processes as needed
        args_list = [(idx, float(ra), float(dec), data_release, radius) for idx, (ra, dec) in enumerate(zip(catalog["RA"], catalog["Dec"]))]
        results = list(tqdm(pool.imap(process_entry, args_list), total=len(args_list)))
    
    for idx, z, zerr, spectype, observed, tileid, petal_loc, targetid, fiber, night in results:
        z_lst.append(z)
        zerr_lst.append(zerr)
        spectype_lst.append(spectype)
        obs_status.append(observed)
        tileid_lst.append(tileid)
        petal_loc_lst.append(petal_loc)
        targetid_lst.append(targetid)
        fiber_lst.append(fiber)
        night_lst.append(night)

    catalog["Z"] = z_lst
    catalog["Z_ERR"] = zerr_lst
    catalog["SPECTYPE"] = spectype_lst
    catalog["OBSERVATION_STATUS"] = obs_status
    catalog["TILEID"] = tileid_lst
    catalog["PETAL_LOC"] = petal_loc_lst
    catalog["TARGETID"] = targetid_lst
    catalog["FIBER"] = fiber_lst
    catalog["NIGHT"] = night_lst

    n_match = len(catalog["Z"]) - catalog["Z"].isna().sum()
    n_tot = len(catalog["Z"])
    print(f"{n_match} out of {n_tot} objects have a DESI redshift.")

    catalog.to_csv(outfile)

if __name__ == "__main__":
    main()




