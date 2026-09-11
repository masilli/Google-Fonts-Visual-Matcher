import os
import urllib.request

# Direct mapping to Google Fonts official GitHub master branch
FONT_FILES = {
    # Workhorse Sans & UI Fonts (incluindo pesos Bold para paridade de peso)
    "Inter-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf",
    "Inter-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf",
    "Roboto-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/roboto/Roboto%5Bwdth%2Cwght%5D.ttf",
    "Roboto-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/roboto/Roboto%5Bwdth%2Cwght%5D.ttf",
    "OpenSans-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/opensans/OpenSans%5Bwdth%2Cwght%5D.ttf",
    "OpenSans-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/opensans/OpenSans%5Bwdth%2Cwght%5D.ttf",
    "Lato-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/lato/Lato-Regular.ttf",
    "Lato-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/lato/Lato-Bold.ttf",
    "NotoSans-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf",
    "NotoSans-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf",
    "WorkSans-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/worksans/WorkSans%5Bwght%5D.ttf",
    "WorkSans-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/worksans/WorkSans%5Bwght%5D.ttf",
    "DM_Sans-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/dmsans/DMSans%5Bopsz%2Cwght%5D.ttf",
    "DM_Sans-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/dmsans/DMSans%5Bopsz%2Cwght%5D.ttf",

    # Geometric & Modern Sans
    "Outfit-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/outfit/Outfit%5Bwght%5D.ttf",
    "Outfit-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/outfit/Outfit%5Bwght%5D.ttf",
    "Lexend-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/lexend/Lexend%5Bwght%5D.ttf",
    "PlusJakartaSans-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/plusjakartasans/PlusJakartaSans%5Bwght%5D.ttf",
    "Poppins-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/poppins/Poppins-Regular.ttf",
    "Montserrat-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf",
    "Raleway-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/raleway/Raleway%5Bwght%5D.ttf",
    "Nunito-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/nunito/Nunito%5Bwght%5D.ttf",
    "SpaceGrotesk-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/spacegrotesk/SpaceGrotesk%5Bwght%5D.ttf",
    "SpaceGrotesk-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/spacegrotesk/SpaceGrotesk%5Bwght%5D.ttf",
    "Manrope-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/manrope/Manrope%5Bwght%5D.ttf",
    "Sora-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/sora/Sora%5Bwght%5D.ttf",
    "Urbanist-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/urbanist/Urbanist%5Bwght%5D.ttf",
    "AlbertSans-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/albertsans/AlbertSans%5Bwght%5D.ttf",

    # Editorial & Classic Serifs
    "Merriweather-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/merriweather/Merriweather%5Bopsz%2Cwdth%2Cwght%5D.ttf",
    "Lora-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/lora/Lora%5Bwght%5D.ttf",
    "Lora-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/lora/Lora%5Bwght%5D.ttf",
    "EB_Garamond-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/ebgaramond/EBGaramond%5Bwght%5D.ttf",
    "Cinzel-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/cinzel/Cinzel%5Bwght%5D.ttf",
    "BodoniModa-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/bodonimoda/BodoniModa%5Bopsz%2Cwght%5D.ttf",
    "CormorantGaramond-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/cormorantgaramond/CormorantGaramond%5Bwght%5D.ttf",
    "PlayfairDisplay-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplay/PlayfairDisplay%5Bwght%5D.ttf",
    "Playfair-DisplaySC.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplaysc/PlayfairDisplaySC-Regular.ttf",
    "PTSerif-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/ptserif/PT_Serif-Web-Regular.ttf",
    "LibreBaskerville-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/librebaskerville/LibreBaskerville%5Bwght%5D.ttf",
    "Prata-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/prata/Prata-Regular.ttf",
    "Castoro-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/castoro/Castoro-Regular.ttf",
    "CinzelDecorative-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/cinzeldecorative/CinzelDecorative-Regular.ttf",
    "Fraunces-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/fraunces/Fraunces%5BSOFT%2CWONK%2Copsz%2Cwght%5D.ttf",
    "DM_Serif_Display-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/dmserifdisplay/DMSerifDisplay-Regular.ttf",

    # Display & Headline Sans (Condensed / Impact)
    "BarlowCondensed-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/barlowcondensed/BarlowCondensed-Regular.ttf",
    "BarlowCondensed-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/barlowcondensed/BarlowCondensed-Bold.ttf",
    "Teko-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/teko/Teko%5Bwght%5D.ttf",
    "Syne-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/syne/Syne%5Bwght%5D.ttf",
    "Syne-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/syne/Syne%5Bwght%5D.ttf",
    "Righteous-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/righteous/Righteous-Regular.ttf",
    "BebasNeue-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/bebasneue/BebasNeue-Regular.ttf",
    "Oswald-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/oswald/Oswald%5Bwght%5D.ttf",
    "Anton-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf",
    "Staatliches-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/staatliches/Staatliches-Regular.ttf",
    "RussoOne-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/russoone/RussoOne-Regular.ttf",
    "BlackOpsOne-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/blackopsone/BlackOpsOne-Regular.ttf",
    "AbrilFatface-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/abrilfatface/AbrilFatface-Regular.ttf",
    "Bungee-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/bungee/Bungee-Regular.ttf",
    "BungeeShade-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/bungeeshade/BungeeShade-Regular.ttf",
    "RubikBeastly-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/rubikbeastly/RubikBeastly-Regular.ttf",
    "AlumniSans-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/alumnisans/AlumniSans%5Bwght%5D.ttf",
    "Chango-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/chango/Chango-Regular.ttf",

    # Script & Handwriting
    "Pacifico-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/pacifico/Pacifico-Regular.ttf",
    "Caveat-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/caveat/Caveat%5Bwght%5D.ttf",

    # Humanist & Rounded Sans
    "Quicksand-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/quicksand/Quicksand%5Bwght%5D.ttf",
    "Comfortaa-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/comfortaa/Comfortaa%5Bwght%5D.ttf",
    "Dosis-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/dosis/Dosis%5Bwght%5D.ttf",
    "VarelaRound-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/varelaround/VarelaRound-Regular.ttf",

    # Slab Serifs
    "RobotoSlab-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/RobotoSlab%5Bwght%5D.ttf",
    "ZillaSlab-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/zillaslab/ZillaSlab-Regular.ttf",
    "Arvo-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/arvo/Arvo-Regular.ttf",
    "CreteRound-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/creteround/CreteRound-Regular.ttf",
    "AlfaSlabOne-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/alfaslabone/AlfaSlabOne-Regular.ttf",

    # Monospaced
    "JetBrainsMono-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf",
    "SpaceMono-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/spacemono/SpaceMono-Regular.ttf",
    "RobotoMono-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/robotomono/RobotoMono%5Bwght%5D.ttf",
    "FiraCode-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/firacode/FiraCode%5Bwght%5D.ttf",
    "Inconsolata-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/inconsolata/Inconsolata%5Bwdth%2Cwght%5D.ttf",
    "SourceCodePro-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/sourcecodepro/SourceCodePro%5Bwght%5D.ttf",
}

def download_fonts(output_dir: str = "./fonts"):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename, url in FONT_FILES.items():
        dest = os.path.join(output_dir, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"Skipping {filename} (already exists)")
            continue
        print(f"Downloading {filename}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(dest, "wb") as f:
                f.write(response.read())
            print(f"Saved {filename}")
        except Exception as e:
            print(f"Failed {filename}: {e}")

if __name__ == "__main__":
    download_fonts()
    print("\nFonts ready in ./fonts.")