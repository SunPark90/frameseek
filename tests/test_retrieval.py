import unittest

from frameseek.models import FrameRecord, VideoIndex, VideoMetadata
from frameseek.retrieval import rank_frames, tokenize


def make_index(captions: list[str | None]) -> VideoIndex:
    return VideoIndex(
        video=VideoMetadata(source="sample.mp4", duration_seconds=100.0),
        frames=tuple(
            FrameRecord(
                id=f"f{number:06d}",
                timestamp_seconds=float(number * 10),
                path=f"frames/{number}.jpg",
                caption=caption,
            )
            for number, caption in enumerate(captions)
        ),
    )


class RetrievalTests(unittest.TestCase):
    def test_tokenizer_supports_korean(self) -> None:
        self.assertEqual(tokenize("강아지가 공원에서 달린다"), ("강아지가", "공원에서", "달린다"))

    def test_caption_overlap_ranks_relevant_frame(self) -> None:
        index = make_index(
            [
                "한 남자가 주방에서 요리한다",
                "검은 강아지가 공원에서 빨간 공을 쫓는다",
                "도시 야경과 자동차가 보인다",
            ]
        )
        ranked = rank_frames(index, "공원에서 빨간 공을 쫓는 장면", top_k=1)
        self.assertEqual(ranked[0].frame.id, "f000001")
        self.assertGreater(ranked[0].score, 0)

    def test_uncaptioned_index_uses_temporal_spread(self) -> None:
        index = make_index([None, None, None, None, None])
        ranked = rank_frames(index, "무슨 일이 일어났나?", top_k=3)
        self.assertEqual([item.frame.id for item in ranked], ["f000000", "f000002", "f000004"])


if __name__ == "__main__":
    unittest.main()
