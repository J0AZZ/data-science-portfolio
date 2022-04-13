import jams

jam_path = 'resources/Koops/jams/591.jams'
#jam_path = 'jams/1235.jams'

jam = jams.load(jam_path)

print(jam)

"annotations": [
    {
      "annotation_metadata": {
        "curator": {
          "name": "",
          "email": ""
        },
        "annotator": {
          "instrument": "Guitar",
          "id": "A1"
        },
        "version": 1,
        "corpus": "",
        "annotation_tools": "https://chordify.net/",
        "annotation_rules": "",
        "validation": "",
        "data_source": "expert human"
      },
      "namespace": "chord",
      "data": [
        {
          "time": 0.0,
          "duration": 0.16253968,
          "value": "N",
          "confidence": 1.0
        },
      ],
      "sandbox": {
        "reported_time": 10,
        "reported_difficulty": 1
      },
      "time": 0,
      "duration": 222.2
    },
   ],
  "file_metadata": {
    "title": "The Long Run",
    "artist": "Eagles",
    "release": "",
    "duration": 222.2,
    "identifiers": {
      "youtube_url": "https://www.youtube.com/watch?v=DI_rkZIuHzw",
      "bbid": 591
    },
    "jams_version": "0.3.0"
  },
  "sandbox": {}
}