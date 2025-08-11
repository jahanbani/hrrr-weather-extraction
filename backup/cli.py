#!/usr/bin/env python3
"""
Command-line interface for HRRR data extraction.
Provides easy-to-use commands for different extraction types.
"""

import click
import datetime
import os
import sys
from pathlib import Path
from typing import Optional

# Import enhanced modules
from config_unified import HRRRConfig, load_config_from_file, save_config_to_file
from utils_enhanced import validate_inputs, log_system_info, check_disk_space
from hrrr_enhanced import extract_specific_locations_enhanced, extract_full_grid_enhanced


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """HRRR Data Extraction Tool
    
    Extract HRRR weather data for wind and solar locations.
    """
    # Set up logging level
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if config:
        try:
            ctx.obj = load_config_from_file(config)
            click.echo(f"✅ Loaded configuration from {config}")
        except Exception as e:
            click.echo(f"❌ Error loading config: {e}", err=True)
            sys.exit(1)
    else:
        ctx.obj = HRRRConfig()
        click.echo("✅ Using default configuration")


@cli.command()
@click.option('--wind-csv', default='wind.csv', help='Wind locations CSV file')
@click.option('--solar-csv', default='solar.csv', help='Solar locations CSV file')
@click.option('--start-date', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', required=True, help='End date (YYYY-MM-DD)')
@click.option('--output-dir', default='./output', help='Output directory')
@click.option('--workers', type=int, help='Number of workers (auto-detect if not specified)')
@click.option('--chunk-size', type=int, default=100, help='Processing chunk size')
@click.option('--no-parallel', is_flag=True, help='Disable parallel processing')
@click.option('--no-resume', is_flag=True, help='Disable resume functionality')
@click.option('--save-config', type=click.Path(), help='Save configuration to file')
@click.pass_context
def extract_specific(ctx, wind_csv, solar_csv, start_date, end_date, output_dir, 
                    workers, chunk_size, no_parallel, no_resume, save_config):
    """Extract HRRR data for specific wind and solar locations."""
    
    config = ctx.obj
    
    # Update configuration with command line options
    config.wind_csv_path = wind_csv
    config.solar_csv_path = solar_csv
    config.output_base_dir = output_dir
    config.use_parallel = not no_parallel
    config.enable_resume = not no_resume
    config.chunk_size = chunk_size
    
    if workers:
        config.num_workers = workers
    
    # Parse dates
    try:
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        config.start_date = start_dt
        config.end_date = end_dt
    except ValueError as e:
        click.echo(f"❌ Invalid date format: {e}", err=True)
        sys.exit(1)
    
    # Validate configuration
    try:
        config.validate()
        click.echo("✅ Configuration validated")
    except Exception as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
        sys.exit(1)
    
    # Save configuration if requested
    if save_config:
        try:
            save_config_to_file(config, save_config)
            click.echo(f"✅ Configuration saved to {save_config}")
        except Exception as e:
            click.echo(f"❌ Error saving config: {e}", err=True)
    
    # Show configuration
    click.echo("\n📋 Extraction Configuration:")
    click.echo(f"   Wind CSV: {config.wind_csv_path}")
    click.echo(f"   Solar CSV: {config.solar_csv_path}")
    click.echo(f"   Date range: {start_date} to {end_date}")
    click.echo(f"   Output directory: {output_dir}")
    click.echo(f"   Workers: {config.num_workers}")
    click.echo(f"   Chunk size: {config.chunk_size}")
    click.echo(f"   Parallel processing: {config.use_parallel}")
    click.echo(f"   Resume enabled: {config.enable_resume}")
    
    # Confirm before proceeding
    if not click.confirm("\nProceed with extraction?"):
        click.echo("❌ Extraction cancelled")
        sys.exit(0)
    
    # Run extraction
    click.echo("\n🚀 Starting extraction...")
    try:
        result = extract_specific_locations_enhanced(config)
        if result:
            click.echo("✅ Extraction completed successfully!")
        else:
            click.echo("❌ Extraction failed!", err=True)
            sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Extraction error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--start-date', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', required=True, help='End date (YYYY-MM-DD)')
@click.option('--output-dir', default='./output', help='Output directory')
@click.option('--workers', type=int, help='Number of workers (auto-detect if not specified)')
@click.option('--chunk-size', type=int, default=100, help='Processing chunk size')
@click.option('--no-parallel', is_flag=True, help='Disable parallel processing')
@click.option('--no-resume', is_flag=True, help='Disable resume functionality')
@click.option('--save-config', type=click.Path(), help='Save configuration to file')
@click.pass_context
def extract_full_grid(ctx, start_date, end_date, output_dir, workers, chunk_size, 
                     no_parallel, no_resume, save_config):
    """Extract HRRR data for the full grid (1.9M+ points)."""
    
    config = ctx.obj
    
    # Update configuration
    config.output_base_dir = output_dir
    config.use_parallel = not no_parallel
    config.enable_resume = not no_resume
    config.chunk_size = chunk_size
    
    if workers:
        config.num_workers = workers
    
    # Parse dates
    try:
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        config.start_date = start_dt
        config.end_date = end_dt
    except ValueError as e:
        click.echo(f"❌ Invalid date format: {e}", err=True)
        sys.exit(1)
    
    # Validate configuration
    try:
        config.validate()
        click.echo("✅ Configuration validated")
    except Exception as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
        sys.exit(1)
    
    # Save configuration if requested
    if save_config:
        try:
            save_config_to_file(config, save_config)
            click.echo(f"✅ Configuration saved to {save_config}")
        except Exception as e:
            click.echo(f"❌ Error saving config: {e}", err=True)
    
    # Show configuration
    click.echo("\n📋 Full Grid Extraction Configuration:")
    click.echo(f"   Date range: {start_date} to {end_date}")
    click.echo(f"   Output directory: {output_dir}")
    click.echo(f"   Workers: {config.num_workers}")
    click.echo(f"   Chunk size: {config.chunk_size}")
    click.echo(f"   Parallel processing: {config.use_parallel}")
    click.echo(f"   Resume enabled: {config.enable_resume}")
    click.echo("⚠️  WARNING: Full grid extraction requires significant resources!")
    
    # Confirm before proceeding
    if not click.confirm("\nProceed with full grid extraction?"):
        click.echo("❌ Extraction cancelled")
        sys.exit(0)
    
    # Run extraction
    click.echo("\n🚀 Starting full grid extraction...")
    try:
        result = extract_full_grid_enhanced(config)
        if result:
            click.echo("✅ Full grid extraction completed successfully!")
        else:
            click.echo("❌ Full grid extraction failed!", err=True)
            sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Extraction error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--wind-csv', default='wind.csv', help='Wind locations CSV file')
@click.option('--solar-csv', default='solar.csv', help='Solar locations CSV file')
def validate(wind_csv, solar_csv):
    """Validate input files and system configuration."""
    
    click.echo("🔍 Validating system and inputs...")
    
    try:
        # Check system info
        log_system_info()
        
        # Validate input files
        validate_inputs(wind_csv, solar_csv)
        
        # Check disk space
        check_disk_space('./output', required_gb=10.0)
        
        click.echo("✅ All validations passed!")
        
    except Exception as e:
        click.echo(f"❌ Validation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', default='config.json', help='Output file path')
def create_config(output):
    """Create a default configuration file."""
    
    try:
        config = HRRRConfig()
        save_config_to_file(config, output)
        click.echo(f"✅ Configuration file created: {output}")
        click.echo("📝 Edit the file to customize settings")
        
    except Exception as e:
        click.echo(f"❌ Error creating config: {e}", err=True)
        sys.exit(1)


@cli.command()
def info():
    """Show system information and configuration."""
    
    click.echo("📊 System Information:")
    log_system_info()
    
    click.echo("\n📋 Default Configuration:")
    config = HRRRConfig()
    click.echo(f"   Wind CSV: {config.wind_csv_path}")
    click.echo(f"   Solar CSV: {config.solar_csv_path}")
    click.echo(f"   Output directory: {config.output_base_dir}")
    click.echo(f"   Workers: {config.num_workers}")
    click.echo(f"   Chunk size: {config.chunk_size}")
    click.echo(f"   Parallel processing: {config.use_parallel}")
    click.echo(f"   Resume enabled: {config.enable_resume}")
    
    # Check if input files exist
    click.echo("\n📁 File Status:")
    for name, path in [("Wind CSV", config.wind_csv_path), 
                      ("Solar CSV", config.solar_csv_path)]:
        if os.path.exists(path):
            click.echo(f"   ✅ {name}: {path}")
        else:
            click.echo(f"   ❌ {name}: {path} (not found)")


if __name__ == '__main__':
    cli() 