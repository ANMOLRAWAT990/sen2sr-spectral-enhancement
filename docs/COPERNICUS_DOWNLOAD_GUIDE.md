# Copernicus Browser: Sentinel-2 10-Band Export Guide
### Step-by-Step Instructions for Multispectral Super-Resolution

This guide details how to find, filter, and export analytical Sentinel-2 Level-2A (L2A) multispectral imagery from the Copernicus Data Space Ecosystem (CDSE) for use with the SEN2SR pipeline.

---

## 1. Access and Account Setup

1. **Website:** Go to [Copernicus Browser](https://browser.dataspace.copernicus.eu/).
2. **Account:** Click **Login** in the upper-right corner. If you do not have an account, register for a free Copernicus Data Space account (registration is free and activates immediately).
3. **Data Source:** In the left search panel, verify that **Sentinel-2** is selected.

---

## 2. Search and Filter for Cloud-Free Scenes

To ensure high-quality super-resolution without cloud artifacts or atmospheric distortion:

1. **Search Parameters:**
   - **Collection:** Select **Sentinel-2 L2A** (Bottom-Of-Atmosphere / Surface Reflectance). Do not use L1C (Top-Of-Atmosphere).
   - **Max Cloud Coverage:** Set the cloud coverage filter to **<= 10%** (or 0-5% for clearest conditions).
   - **Time Range:** Select a recent date range (e.g., past 1-3 months) or specific seasonal dates (e.g., post-monsoon for green vegetation).
2. **Define Area of Interest (AOI):**
   - Use the map to navigate to your target region (e.g., riverbanks, farm parcels, urban areas).
   - Click the **Square / Polygon Tool** on the right toolbar and draw a box around your target study area.
3. **Select a Scene:**
   - Click **Search**.
   - Browse the results and click **Visualize** on a scene that is clear and free of clouds over the study area.

---

## 3. Download Configuration

Click the **Download Image** icon on the right toolbar. In the download window, set the following parameters:

### A. Format and Quality Settings
| Option | Setting | Reason |
| :--- | :--- | :--- |
| **Image Format** | **TIFF (16-bit)** | Native Sentinel-2 integer format (scaled by 10,000). Preserves full radiometric precision. |
| **Image Resolution** | **HIGH** (~1000x1000 px) | Important: Do not select LOW or MEDIUM, which downsamples pixels. HIGH keeps native 10m spatial detail. |
| **Coordinate System** | **WGS 84 (EPSG:4326)** | Standard global projection used across GIS tools and notebooks. |
| **Clip extra bands** | **ON (Enabled)** | Clips the exported GeoTIFFs to the drawn bounding box. |

---

### B. Layer Selection: 10 Analytical Bands

The SEN2SR pipeline requires 10 spectral bands. Set up the layers as follows:

#### Under "Visualised" (Left Column):
- Uncheck all options (uncheck True color, False color, NDVI, etc.). These are 8-bit display images and cannot be used for scientific model inference.

#### Under "Raw" (Right Column):
- Click the blue **Show more** button at the bottom of the list to reveal bands B11 and B12.
- Check these exact 10 bands:

| Band | Name | Native Resolution | Role in SEN2SR |
| :---: | :--- | :---: | :--- |
| **B02** | Blue | 10m | RGB Composite & 4x Super-Resolution |
| **B03** | Green | 10m | RGB Composite, NDWI (Water Index) |
| **B04** | Red | 10m | RGB Composite, NDVI (Vegetation Index) |
| **B05** | Vegetation Red Edge 1 | 20m | Red Edge Guidance & SWIR Enhancement |
| **B06** | Vegetation Red Edge 2 | 20m | Red Edge Guidance & SWIR Enhancement |
| **B07** | Vegetation Red Edge 3 | 20m | Red Edge Guidance & SWIR Enhancement |
| **B08** | NIR (Broad) | 10m | RGBN Generator, NDVI & NDWI Computation |
| **B8A** | Narrow NIR | 20m | Red Edge & Vegetation Biophysical Assessment |
| **B11** | SWIR-1 | 20m | NDBI (Built-Up Index) & Soil Moisture |
| **B12** | SWIR-2 | 20m | Geology & Soil Composition |

*(Leave B01 and B09 unchecked - these are 60m atmospheric correction bands not used in surface mapping).*

---

## 4. Saving and Using with the Notebooks

1. Click **Download** at the bottom of the window.
2. The browser will download a single compressed ZIP archive (for example, `Browser_images.zip` or `Browser_images_1.zip`).
3. **Upload to Google Drive:**
   Place the downloaded ZIP file into your project folder:
   ```text
   My Drive/
   └── Project_Folder/
       └── Datasets/
           └── Browser_images_1.zip
   ```
4. **Run the Notebook:**
   Open `notebooks/04_finetuning_experiments/SEN2SR_Baseline_vs_Finetuned_Comparison.ipynb` in Google Colab.
   - The notebook automatically detects and extracts the ZIP archive.
   - It aligns the 10 bands into an analytical stack.
   - It runs baseline and fine-tuned models, displaying side-by-side images, difference heatmaps, and spectral index metrics.

---

## 5. Sample Indian Coordinates for Testing

If looking for test regions with varied landscapes across India:

| Region | Coordinates / Search | Features |
| :--- | :--- | :--- |
| **Varanasi / Ganga River** | `25.3176 N, 82.9739 E` | Riverbanks, water-land interface (NDWI), and dense urban ghats (NDBI). |
| **Punjab Agricultural Belt** | `30.9010 N, 75.8573 E` (Ludhiana) | Clear field boundaries and crop parcel transitions (NDVI). |
| **Delhi-NCR Urban Belt** | `28.6139 N, 77.2090 E` | Dense road networks, bridges, and Yamuna floodplains. |
| **Western Ghats** | `10.8505 N, 76.2711 E` | Forest canopy, mountain contours, and reservoir edges. |
| **Sundarbans** | `21.9497 N, 89.1833 E` | Tidal estuarine channels and mangrove borders. |

---

## 6. Troubleshooting FAQ

### Why is the downloaded image small (e.g. 52x36 pixels)?
The map was zoomed out too far, or Image Resolution was left on MEDIUM or LOW. Zoom into your target area on the map so the box covers a local area, and verify that Image Resolution is set to **HIGH**.

### Why does the notebook show "ValueError: Missing required bands"?
Bands B11 and B12 were not checked. In Copernicus Browser, click "Show more" under Raw layers to reveal B11 and B12, and ensure both are included in the download.

### Should the ZIP file be extracted before uploading to Drive?
No, keep it as a `.zip` archive. The Colab notebook extracts the archive directly into local `/content` storage, which is much faster than uploading hundreds of separate files.

### Why not download PNG or JPEG?
PNG and JPEG are 8-bit images (0-255) made for screen display. They discard the underlying multispectral reflectance data. The super-resolution model requires raw surface reflectance values, which are preserved only in 16-bit GeoTIFF format.
