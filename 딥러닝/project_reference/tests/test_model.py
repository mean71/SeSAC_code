# tests/test_model.py

import unittest  # unittest 모듈을 임포트하여 테스트 프레임워크를 사용
from src.train import ModelA, ModelB  # 학습할 모델 클래스를 임포트
from src.data_loader import load_data, preprocess_data  # 데이터 로딩 및 전처리 함수 임포트

class TestModels(unittest.TestCase):
    """
    모델을 테스트하기 위한 유닛 테스트 클래스
    unittest.TestCase를 상속받아 각 모델의 학습 및 예측 기능을 테스트
    """

    def setUp(self):
        """
        각 테스트 실행 전에 필요한 설정을 초기화하는 메서드
        테스트가 실행되기 전에 항상 호출되어 데이터 로드 및 모델 인스턴스를 생성
        """
        # 데이터 로드: 원본 데이터를 지정된 경로에서 불러옴
        self.raw_data = load_data('../data/raw/train_data.csv')
        # 전처리: 로드한 데이터를 모델 A에 맞게 전처리
        self.processed_data = preprocess_data(self.raw_data, 'model_a')

        # 모델 인스턴스 생성: 각각의 모델 클래스의 인스턴스를 생성
        self.model_a = ModelA()  # 모델 A 인스턴스
        self.model_b = ModelB()  # 모델 B 인스턴스

    def test_model_a_training(self):
        """
        모델 A의 학습 기능을 테스트하는 메서드
        학습이 정상적으로 진행되는지를 검증
        """
        try:
            self.model_a.train(self.processed_data)  # 모델 A를 전처리된 데이터로 학습
            self.assertTrue(True, "모델 A 학습 성공")  # 학습이 성공하면 True 반환
        except Exception as e:
            self.fail(f"모델 A 학습 실패: {str(e)}")  # 예외 발생 시 실패 메시지 출력

    def test_model_b_training(self):
        """
        모델 B의 학습 기능을 테스트하는 메서드
        학습이 정상적으로 진행되는지를 검증
        """
        try:
            self.model_b.train(self.processed_data)  # 모델 B를 전처리된 데이터로 학습
            self.assertTrue(True, "모델 B 학습 성공")  # 학습이 성공하면 True 반환
        except Exception as e:
            self.fail(f"모델 B 학습 실패: {str(e)}")  # 예외 발생 시 실패 메시지 출력

    def test_model_a_inference(self):
        """
        모델 A의 추론 기능을 테스트하는 메서드
        모델 A가 정상적으로 예측을 수행하는지를 검증
        """
        try:
            predictions = self.model_a.predict(self.processed_data)  # 모델 A로부터 예측값을 가져옴
            self.assertIsNotNone(predictions, "모델 A 예측 성공")  # 예측 결과가 None이 아니어야 함
        except Exception as e:
            self.fail(f"모델 A 예측 실패: {str(e)}")  # 예외 발생 시 실패 메시지 출력

    def test_model_b_inference(self):
        """
        모델 B의 추론 기능을 테스트하는 메서드
        모델 B가 정상적으로 예측을 수행하는지를 검증
        """
        try:
            predictions = self.model_b.predict(self.processed_data)  # 모델 B로부터 예측값을 가져옴
            self.assertIsNotNone(predictions, "모델 B 예측 성공")  # 예측 결과가 None이 아니어야 함
        except Exception as e:
            self.fail(f"모델 B 예측 실패: {str(e)}")  # 예외 발생 시 실패 메시지 출력

if __name__ == "__main__":
    unittest.main()  # 이 파일이 직접 실행될 경우 모든 테스트를 실행
