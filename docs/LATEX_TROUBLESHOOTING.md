# PRiSM Pipeline LaTeX Troubleshooting Guide

This guide helps you resolve LaTeX-related errors when running the PRiSM pipeline script (`run_prism_pipeline.py`). The pipeline generates PDF reports from Jupyter notebooks, which requires LaTeX to be properly installed and configured.

## Prerequisites

This guide assumes you have already:
- ✅ Installed Python 3.11+ and created the `venv_prism` virtual environment
- ✅ Installed all Python dependencies from `requirements.txt`
- ✅ Successfully activated your virtual environment

## Common LaTeX Errors and Solutions

### 1. "Pandoc wasn't found" Error

**Error Message:**
```
nbconvert.utils.pandoc.PandocMissing: Pandoc wasn't found.
Please check that pandoc is installed:
https://pandoc.org/installing.html
```

**Solution:**

**macOS:**
```bash
brew install pandoc
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install pandoc
```

**Linux (CentOS/RHEL):**
```bash
sudo yum install pandoc
```

**Windows:**
- Download from [pandoc.org](https://pandoc.org/installing.html)
- Or use Chocolatey: `choco install pandoc`

**Verify Installation:**
```bash
pandoc --version
```

### 2. "xelatex not found" Error

**Error Message:**
```
OSError: xelatex not found on PATH, if you have not installed xelatex you may need to do so.
```

**Solution:**

**macOS:**
```bash
# Install BasicTeX (minimal LaTeX distribution)
brew install --cask basictex

# Update PATH to include TeX binaries
eval "$(/usr/libexec/path_helper)"

# Verify installation
xelatex --version
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install texlive-xetex texlive-latex-base texlive-latex-extra
```

**Linux (CentOS/RHEL):**
```bash
sudo yum install texlive-xetex texlive-latex texlive-latex-extra
```

**Windows:**
- Install MiKTeX from [miktex.org](https://miktex.org/download)
- Choose "Install missing packages on-the-fly: Yes" during installation

### 3. Missing LaTeX Package Errors

**Error Messages:**
```
! LaTeX Error: File 'tcolorbox.sty' not found.
! LaTeX Error: File 'pdfcol.sty' not found.
! LaTeX Error: File 'soul.sty' not found.
! LaTeX Error: File 'rsfs10' not found.
```

**Solution:**

**macOS (with BasicTeX):**
```bash
# Update PATH first
eval "$(/usr/libexec/path_helper)"

# Install essential packages for Jupyter notebook PDF conversion
sudo tlmgr install tcolorbox pdfcol soul rsfs adjustbox collectbox enumitem makecell multirow tabu threeparttable varwidth

# Update package database
sudo mktexlsr
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install texlive-science texlive-publishers texlive-fonts-recommended
```

**Linux (CentOS/RHEL):**
```bash
sudo yum install texlive-science texlive-publishers texlive-fonts-recommended
```

**Windows (MiKTeX):**
- Open MiKTeX Console
- Go to "Packages" tab
- Search for and install these packages:
  - `tcolorbox`
  - `pdfcol`
  - `soul`
  - `rsfs`
  - `adjustbox`
  - `collectbox`
  - `enumitem`
  - `makecell`
  - `multirow`
  - `tabu`
  - `threeparttable`
  - `varwidth`

### 4. Font File Not Found Errors

**Error Messages:**
```
! I can't find file `rsfs10'.
! I can't find file `cmr10'.
```

**Solution:**

**macOS:**
```bash
# Install font packages
sudo tlmgr install rsfs cm-super

# Update font database
sudo mktexlsr
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install texlive-fonts-recommended texlive-fonts-extra

# CentOS/RHEL
sudo yum install texlive-fonts-recommended texlive-fonts-extra
```

### 5. Permission Errors

**Error Messages:**
```
You don't have permission to change the installation
tlmgr: package repository ... (verified)
```

**Solution:**

**macOS/Linux:**
```bash
# Use sudo for system-wide installation
sudo tlmgr install [package-name]

# Or install to user directory (if you prefer)
tlmgr install --user [package-name]
```

**Windows:**
- Run MiKTeX Console as Administrator
- Or use MiKTeX Console's user package installation

## Complete LaTeX Setup Commands

### macOS (Complete Setup)
```bash
# Install Pandoc
brew install pandoc

# Install BasicTeX
brew install --cask basictex

# Update PATH
eval "$(/usr/libexec/path_helper)"

# Install all required packages
sudo tlmgr install tcolorbox pdfcol soul rsfs adjustbox collectbox enumitem makecell multirow tabu threeparttable varwidth cm-super

# Update databases
sudo mktexlsr

# Verify installation
pandoc --version
xelatex --version
kpsewhich tcolorbox.sty
```

### Linux Ubuntu/Debian (Complete Setup)
```bash
# Install all LaTeX components
sudo apt-get update
sudo apt-get install pandoc texlive-xetex texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-science texlive-publishers

# Verify installation
pandoc --version
xelatex --version
```

### Linux CentOS/RHEL (Complete Setup)
```bash
# Install all LaTeX components
sudo yum install pandoc texlive-xetex texlive-latex texlive-latex-extra texlive-fonts-recommended texlive-science texlive-publishers

# Verify installation
pandoc --version
xelatex --version
```

## Testing Your LaTeX Installation

After installation, test your setup:

```bash
# Test Pandoc
pandoc --version

# Test XeLaTeX
xelatex --version

# Test package availability
kpsewhich tcolorbox.sty
kpsewhich pdfcol.sty
kpsewhich soul.sty

# Test LaTeX compilation
echo '\documentclass{article}\usepackage{tcolorbox}\begin{document}Test\end{document}' > test.tex
xelatex test.tex
rm test.tex test.pdf test.log test.aux
```

## Running the Pipeline

Once LaTeX is properly installed, run the pipeline:

```bash
# Activate your virtual environment
source venv_prism/bin/activate  # macOS/Linux
# OR
venv_prism\Scripts\activate     # Windows

# Run the pipeline
python run_prism_pipeline.py htx_example
```

## Troubleshooting Tips

1. **Restart Terminal**: After installing LaTeX packages, restart your terminal to ensure PATH updates take effect.

2. **Check PATH**: Ensure LaTeX binaries are in your PATH:
   ```bash
   which xelatex
   which pandoc
   ```

3. **Update Package Database**: After installing packages, always run:
   ```bash
   sudo mktexlsr  # macOS/Linux
   ```

4. **Virtual Environment**: Make sure you're running the pipeline with your virtual environment activated.

5. **Package Verification**: Use `kpsewhich` to check if packages are found:
   ```bash
   kpsewhich tcolorbox.sty
   ```

## Alternative: Use Full TeX Live Distribution

If you continue having issues with BasicTeX, consider installing the full TeX Live distribution:

**macOS:**
```bash
brew install --cask mactex
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install texlive-full

# CentOS/RHEL
sudo yum install texlive-full
```

**Note**: Full TeX Live is much larger (~5GB) but includes all packages by default.

## Getting Help

If you're still experiencing issues:

1. Check the main [README.md](../README.md) for general setup information
2. Review platform-specific setup guides:
   - [macOS Setup](SETUP_MACOS.md)
   - [Linux Setup](SETUP_LINUX.md)
   - [Windows Setup](SETUP_WINDOWS.md)
3. Open an issue on GitHub with:
   - Your operating system
   - The exact error message
   - Steps you've already tried
