# GPOST
## General Purpose On-Screen Translator  
This is a tool that allows you to automatically detect, translate and overlay translation right on top of the original text bodies.  
ATTENTION: Requires Gemini API key.
![изображение](https://github.com/user-attachments/assets/9b3d0e90-7841-447d-92c9-ffaf367e8bf3)
(Currently text handling is very basic, and often having hard tome conforming to polygon areas)
## Installation  
1. Navigate to folder you want to install GPOST to
2. Open CMD or Powershell
3. Use `git clone https://github.com/Anzhc/GPOST`
4. Run `setup.bat`
5. Launch GPOST with `run_gpost.bat`
6. It will ask you for Gemini API key. Currently i support only Gemini, so you'll have to get it.
You can ignore it and launch program to see how it works, but you would not be able to receive translations.


That should take care of everything.  
GPOST automatically checks for new base YOLO models from my huggingface repo.  

## How to use
I would recommend binding 3 shortcuts to either your mouse, or hotkey, this will significantly enhance your experience.  
You need just 3 buttons: `Select Sub-Area`, `Run Clean - Inference - Translate` and `Clear Overlays`  
Running inference queues YOLO for detection. It will try to detect text classes it was trained on in selected area.  
Translate will send it to Gemini. Once we receive response - it will be overlayed on top of original text. If we do not receive it, or there is an error - you will see it in Translation Output window.  

There are multiple various functions that allow you to tweak performance of program, but those 3 buttons are all you need to start.
![изображение](https://github.com/user-attachments/assets/78b83023-1de8-458d-a67d-9afa45f94d40)

## How it works
I utilize YOLO models for detection and segmentation, which then crop areas that require translation and send those chunks to Gemini. Then we read json that Gemini returns(if any), and overlay it on top of original text.  
I have added UI section that allows user to filter classes they want to translate, and which should be skipped. Those classes are populated straight from models loaded.  
If any of the TTS are selected, translated text will be sent for voiceover. Once we receive it, it is played and then saved to TTS folder, for future listening, if needed(But i think i forgot to add saving to 11Labs TTS).
