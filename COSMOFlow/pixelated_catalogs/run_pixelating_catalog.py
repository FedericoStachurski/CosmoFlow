import argparse
from galaxy_catalog_handle import GladePlusCatalog

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process galaxy catalog data.")
    parser.add_argument('--catalog_name', type=str, help='Name of the galaxy catalog (e.g., "GLADE+").')
    parser.add_argument('--file_path', type=str, help='Path to the data file.')
    parser.add_argument('--output_directory', type=str, help='Directory where pixel files will be saved.')
    parser.add_argument('--nside', type=int, default=128, help='Healpix NSIDE value (default: 128)')
    parser.add_argument('--chunksize', type=int, default=int(1e6), help='Chunk size for reading data (default: 1e6)')
    parser.add_argument('--Pool', type=int, default=int(10), help='Multiprocessing Pool')
    args = parser.parse_args()
    
    if args.catalog_name == "GLADE+":
        glade_catalog = GladePlusCatalog(args.file_path, nside=args.nside, chunksize=args.chunksize)
    else:
        raise ValueError(f"Unknown catalog name: {args.catalog_name}")
    
    # Load the data
    glade_catalog.load_data()
    
    # Apply K-corrections
    glade_catalog.apply_k_corrections()
    
    # Assign Healpix pixel indices
    glade_catalog.assign_pixels()
    
    # Save pixels to individual files
    glade_catalog.save_pixels(args.output_directory)