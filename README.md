# Table Image to Text

A Python script to extract text from images of document tables. (not furniture)

## How does it work?
1. Using [OpenCV](https://github.com/opencv/opencv/) all the squares in the image are identified.
2. Using these squares, the cells of the table are estimated.
3. The table in the image is chopped into individual images of each cell.
4. Using [EasyOCR](https://github.com/jaidedai/easyocr) the text in each cell is identified.
5. The text is reassembled into a HTML or Markdown table.

The code is largely self-explanatory but feel free to raise an issue if you have a question.

## How well does it work?
I mean, it's okay for a cumulative 14 hours of effort over two days. You can take a look at the samples to get a better idea. 

I know both the samples are images of digital tables but the script works for scanned images as well. That's what I initially wrote the script for. Unfortunately the document I was working on cannot be shared publicly and I couldn't find an alternative.

Keep in mind though, for scanned images, anything below 150dpi might cause issues with OpenCV and EasyOCR (the packages doing the heavy lifting here).

Also, this doesn't work at all for tables with merged cells, vertical text or no borders.

## How can I use it?
I would recommend you set up a virtual environment since there are a bunch of dependencies. (There's a few extra ones in the requirements, I'll clean it up later.)

Download this repository and extract it in a folder. Open a Terminal window and `cd` into the directory. Then install dependencies -

```python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Now you can run the script through the command line like this -
```python
python3 table_image_to_text.py -f <PATH TO IMAGE>
```

This will output the result in the same directory as `output.html` or `output.md`.

### Optional parameters

1. Output: HTML or Markdown. Use the `-o` flag to set it to `html`, `md` or `both`.
2. Headers: Set if there is a header row in the table. Use the `-H` flag to set it to `y` or `n`.
3. View estimated cells: Show the cells estimated by the script. Use the `-c` flag to set it to `y` or `n`.

### Alternative 
You could use the Jupyter notebook if that's more your thing. Set up the virtual environment first though.

## Future work
In order of feasibility:

1. Make it efficient to work with multiple images. Load the OpenCV reader once and stitch output together.
2. Make it work for multiple languages. Just need to add one flag to the argparser. Someone make a PR plz.
3. Make it work for tables with merged cells. That's more on OpenCV but let's see.
4. Make it work for low-resolution images.
5. Make it work for tables without borders. (That's a non-profit I can get behind)

## License
No license. Have fun with it. If you manage to make a commercial product out of this, please let me know. I'd like to worship you.

## Note 
Cheers to the makers and contributors to OpenCV, EasyOCR and PIL. Cheers to open source. Cheers to Python.