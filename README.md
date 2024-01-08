# text_to_triplet

텍스트 데이터를 가공하고 온톨로지 정보에 맞는 triplet으로 변환하는 코드입니다.


`python main.py --input_path ./input --output_path ./output --tasks preprocess filter`

위와 같이 원하는 function을 나열하여 실행하도록 cli에서 조절할 수 있습니다.

`!pip install transformers datasets sentence_transformers peft optimum`