#!/usr/bin/env python3
"""
Script to replace the extract_rectangular_region_day_by_day function with a simplified version
that learns from the successful extract_full_grid_day_by_day approach.
"""

import re

def replace_function():
    # Read the original file
    with open('prereise/gather/winddata/hrrr/calculations.py', 'r') as f:
        content = f.read()
    
    # Define the new function
    new_function = '''def extract_rectangular_region_day_by_day(
    min_lat, max_lat, min_lon, max_lon,
    START,
    END,
    DATADIR,
    DEFAULT_HOURS_FORECASTED,
    SELECTORS,
    output_dir="./extracted_region_data",
    chunk_size=None,  # Auto-detect based on system
    compression="snappy",
    use_parallel=True,
    num_cpu_workers=None,  # Auto-detect based on system
    num_io_workers=None,   # Auto-detect based on system
    max_file_groups=None,  # Auto-detect based on system
    create_individual_mappings=False,
    parallel_file_writing=True,
    enable_resume=True,
    day_output_dir_format="flat",  # "daily" or "flat"
    use_aggressive_settings=True,  # Use aggressive settings for high-performance systems
):
    """
    Extract rectangular region data day by day using the proven full grid approach.
    
    This function uses the same successful approach as extract_full_grid_day_by_day
    but filters grid points to only include those within the specified rectangular region.
    
    Args:
        min_lat, max_lat, min_lon, max_lon (float): Region bounds
        START (datetime): Start datetime
        END (datetime): End datetime
        DATADIR (str): Directory containing GRIB files
        DEFAULT_HOURS_FORECASTED (list): List of forecast hours to process
        SELECTORS (dict): Dictionary of variables to extract
        output_dir (str): Base output directory
        chunk_size (int): Number of grid points per chunk (auto-detect if None)
        compression (str): Parquet compression
        use_parallel (bool): Whether to use parallel processing
        num_cpu_workers (int): Number of CPU workers (auto-detect if None)
        num_io_workers (int): Number of I/O workers (auto-detect if None)
        max_file_groups (int): Maximum file groups to process (auto-detect if None)
        create_individual_mappings (bool): Whether to create individual mapping files
        parallel_file_writing (bool): Whether to use parallel file writing
        enable_resume (bool): Whether to enable resume functionality
        day_output_dir_format (str): Output directory format ("daily" or "flat")
        use_aggressive_settings (bool): Use aggressive settings for high-performance systems
        
    Returns:
        dict: Summary of processing results
    """
    import datetime
    
    print("🚀 Starting Day-by-Day Rectangular Region Extraction (SIMPLIFIED)")
    print("=" * 60)
    print(f"Region: {min_lat}°N to {max_lat}°N, {min_lon}°E to {max_lon}°E")
    print(f"Date range: {START.date()} to {END.date()}")
    print(f"Total days: {(END.date() - START.date()).days + 1}")
    print(f"Output directory: {output_dir}")
    print(f"Day output format: {day_output_dir_format}")
    print(f"Resume enabled: {enable_resume}")
    print(f"Aggressive settings: {use_aggressive_settings}")
    print()
    
    # Auto-detect optimal settings based on system capabilities
    if use_aggressive_settings:
        settings = get_aggressive_parallel_settings()
        
        # Override with user-provided values if specified
        if chunk_size is None:
            chunk_size = settings['chunk_size']
        if num_cpu_workers is None:
            num_cpu_workers = settings['num_cpu_workers']
        if num_io_workers is None:
            num_io_workers = settings['num_io_workers']
        if max_file_groups is None:
            max_file_groups = settings['max_file_groups']
        
        print(f"🎯 Using auto-detected settings:")
        print(f"   Chunk size: {chunk_size:,}")
        print(f"   CPU workers: {num_cpu_workers}")
        print(f"   I/O workers: {num_io_workers}")
        print(f"   Max file groups: {max_file_groups:,}")
        print()
    else:
        # Use conservative defaults if aggressive settings disabled
        if chunk_size is None:
            chunk_size = 150000
        if num_cpu_workers is None:
            num_cpu_workers = 8
        if num_io_workers is None:
            num_io_workers = 4
        if max_file_groups is None:
            max_file_groups = 5000
    
    # Create base output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 🚀 STEP 1: Extract grid metadata (same as full grid)
    print("📊 Extracting grid metadata (once for all days)...")
    grid_start = time.time()
    wind_data_lat_long = get_wind_data_lat_long(START, DATADIR)
    grid_lats, grid_lons = wind_data_lat_long[0], wind_data_lat_long[1]
    
    n_lats, n_lons = grid_lats.shape
    total_grid_points = n_lats * n_lons
    
    print(f"Grid dimensions: {n_lats} x {n_lons} = {total_grid_points:,} total points")
    
    # 🚀 STEP 2: Filter grid points for rectangular region (KEY CHANGE)
    print("🔍 Filtering grid points for rectangular region...")
    region_mask = (grid_lats >= min_lat) & (grid_lats <= max_lat) & \\
                  (grid_lons >= min_lon) & (grid_lons <= max_lon)
    
    region_indices = np.where(region_mask)
    region_lats = grid_lats[region_indices]
    region_lons = grid_lons[region_indices]
    
    # Convert 2D indices to 1D flattened indices for direct GRIB data access
    region_grid_indices = np.ravel_multi_index(region_indices, grid_lats.shape)
    
    print(f"Region points: {len(region_lats):,} out of {total_grid_points:,} total points")
    print(f"Region coverage: {len(region_lats)/total_grid_points*100:.1f}% of full grid")
    
    grid_time = time.time() - grid_start
    print(f"Grid metadata extracted in {grid_time:.2f}s")
    
    # 🚀 STEP 3: Create region mapping (same approach as full grid)
    print("📊 Creating region grid mapping...")
    mapping_start = time.time()
    
    # Create region points DataFrame
    region_points = pd.DataFrame({
        'lat': region_lats,
        'lon': region_lons,
        'grid_index': np.arange(len(region_lats)),  # Sequential index for the region
        'original_grid_index': region_grid_indices  # Original flattened grid indices for direct GRIB access
    })
    
    # Save region mapping
    region_mapping_file = os.path.join(output_dir, "region_mapping.parquet")
    region_points.to_parquet(region_mapping_file, index=False)
    print(f"Region mapping saved to: {region_mapping_file}")
    
    mapping_time = time.time() - mapping_start
    print(f"Region mapping created in {mapping_time:.2f}s")
    
    # 🚀 STEP 4: Initialize tracking (same as full grid)
    total_start_time = time.time()
    successful_days = []
    failed_days = []
    skipped_days = []
    
    # Generate list of days to process
    date_range = pd.date_range(start=START.date(), end=END.date(), freq="1D")
    
    # Check for already processed days if resume is enabled
    if enable_resume:
        print("Checking for already processed days...")
        processed_dates = get_processed_date_range(output_dir, SELECTORS)
        remaining_dates = [d for d in date_range if d.date() not in processed_dates]
        
        if processed_dates:
            print(f"Found {len(processed_dates)} already processed days:")
            for date in sorted(processed_dates):
                print(f"  - {date}")
            print(f"Remaining days to process: {len(remaining_dates)}")
        else:
            print("No previously processed days found.")
            remaining_dates = date_range
    else:
        remaining_dates = date_range
    
    if len(remaining_dates) == 0:
        print("✅ All days already processed! Extraction complete.")
        return {
            "status": "completed",
            "total_days": len(date_range),
            "successful_days": len(successful_days),
            "failed_days": len(failed_days),
            "skipped_days": len(skipped_days),
            "processing_time_seconds": 0,
            "region_points": len(region_lats),
            "resume_used": True
        }
    
    # 🚀 STEP 5: Process each day (same approach as full grid but with region indices)
    print(f"\\n🎯 Processing {len(remaining_dates)} days...")
    
    for day_idx, current_date in enumerate(remaining_dates):
        day_start_time = time.time()
        
        print(f"\\n📅 Processing day {day_idx + 1}/{len(remaining_dates)}: {current_date.date()}")
        
        # Check for shutdown request
        if check_shutdown_requested():
            print("🛑 Shutdown requested. Stopping processing...")
            break
        
        try:
            # Process this day using the same approach as full grid but with region indices
            day_result = process_day_for_rectangular_region(
                current_date=current_date,
                DATADIR=DATADIR,
                DEFAULT_HOURS_FORECASTED=DEFAULT_HOURS_FORECASTED,
                SELECTORS=SELECTORS,
                region_points=region_points,
                output_dir=output_dir,
                compression=compression,
                use_parallel=use_parallel,
                num_cpu_workers=num_cpu_workers,
                num_io_workers=num_io_workers,
                max_file_groups=max_file_groups,
                parallel_file_writing=parallel_file_writing,
                day_output_dir_format=day_output_dir_format
            )
            
            if day_result:
                successful_days.append(current_date.date())
                day_time = time.time() - day_start_time
                print(f"✅ Day {current_date.date()} completed in {day_time:.2f}s")
            else:
                failed_days.append(current_date.date())
                print(f"❌ Day {current_date.date()} failed")
                
        except Exception as e:
            failed_days.append(current_date.date())
            print(f"❌ Error processing day {current_date.date()}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 🚀 STEP 6: Final summary (same as full grid)
    total_time = time.time() - total_start_time
    
    print(f"\\n{'='*60}")
    print(f"RECTANGULAR REGION EXTRACTION COMPLETED")
    print(f"{'='*60}")
    print(f"Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Output directory: {output_dir}")
    print(f"Region mapping: {region_mapping_file}")
    
    print(f"\\n📊 RESULTS SUMMARY:")
    print(f"   Total days: {len(date_range)}")
    print(f"   Successful days: {len(successful_days)}")
    print(f"   Failed days: {len(failed_days)}")
    print(f"   Skipped days: {len(skipped_days)}")
    print(f"   Success rate: {len(successful_days)/len(date_range)*100:.1f}%")
    print(f"   Region points: {len(region_lats):,}")
    print(f"   Region coverage: {len(region_lats)/total_grid_points*100:.1f}% of full grid")
    
    if successful_days:
        avg_day_time = total_time / len(successful_days)
        print(f"   Average time per day: {avg_day_time:.2f}s")
    
    if failed_days:
        print(f"\\n❌ Failed days:")
        for date in failed_days:
            print(f"   - {date}")
    
    # Update resume metadata
    if enable_resume:
        create_resume_metadata(output_dir, START, END, SELECTORS, successful_days)
        print(f"\\n✅ Resume metadata updated")
    
    return {
        "status": "completed" if len(failed_days) == 0 else "completed_with_errors",
        "total_days": len(date_range),
        "successful_days": len(successful_days),
        "failed_days": len(failed_days),
        "skipped_days": len(skipped_days),
        "processing_time_seconds": total_time,
        "region_points": len(region_lats),
        "resume_used": enable_resume and len(processed_dates) > 0 if 'processed_dates' in locals() else False
    }'''
    
    # Find the start and end of the old function
    start_pattern = r'def extract_rectangular_region_day_by_day\('
    end_pattern = r'def process_variable_parallel_rectangular\('
    
    start_match = re.search(start_pattern, content)
    end_match = re.search(end_pattern, content)
    
    if start_match and end_match:
        start_pos = start_match.start()
        end_pos = end_match.start()
        
        # Replace the old function with the new one
        new_content = content[:start_pos] + new_function + '\n\n' + content[end_pos:]
        
        # Write the updated content back to the file
        with open('prereise/gather/winddata/hrrr/calculations.py', 'w') as f:
            f.write(new_content)
        
        print("✅ Successfully replaced the extract_rectangular_region_day_by_day function!")
        print("The new function uses the proven full grid approach but only processes region points.")
    else:
        print("❌ Could not find the function boundaries in the file.")

if __name__ == "__main__":
    replace_function()
