# Maps Directory

This directory is for storing OpenStreetMap (.osm) files.

## How to Get Map Data

### Option 1: OpenStreetMap Website (Recommended for small areas)

1. Go to [OpenStreetMap](https://www.openstreetmap.org/export)
2. Navigate to your area of interest (e.g., Belagavi, India)
3. Click "Manually select a different area" and draw a box
4. Click "Export" to download the .osm file
5. Save it in this directory

### Option 2: Overpass API (For larger areas)

```bash
# Example for Belagavi area
wget -O belagavi_map.osm "https://overpass-api.de/api/map?bbox=74.45,15.82,74.55,15.92"
```

### Option 3: Geofabrik (For regional data)

Visit [Geofabrik Downloads](https://download.geofabrik.de/) for pre-processed regional extracts.

## Converting OSM to SUMO Format

After placing your .osm file here, run:

```bash
python ../scripts/map_converter.py --input belagavi_map.osm --output ../sumo_config/
```

This will generate all necessary SUMO configuration files.

## Notes

- Keep file sizes manageable (< 100MB for reasonable simulation times)
- For very large areas, consider extracting a subset
- Ensure the area includes intersections with traffic signals for meaningful simulation
