from dtos import SearchRequestDTO, SearchResponseDTO, CarSearchResultDTO
from embedder_layers import DINOv2EmbeddingCompressor
from vehicle_comparison_llm import VehicleComparisonLLM
from qdrant_retrieval import QdrantRetrieval
from PIL.Image import Image


class VehicleSearchRag:

    def __init__(self,
        qdrant_api_key: str, qdrant_url: str, collection_name: str,
        model_name: str = "facebook/dinov2-base",
        compressor_model_path: str = "compressor_model.pt",
        llm_model_path: str = "./models"
    ):

        self.embedder = DINOv2EmbeddingCompressor(
            model_name=model_name,
            compressor_model_path=compressor_model_path,
        )

        self.retrieval = QdrantRetrieval(
            api_key=qdrant_api_key,
            url=qdrant_url,
            collection_name=collection_name
        )

        self.llm = VehicleComparisonLLM(
            additional_weights_path=llm_model_path
        )


    def __call__(self, imgs: list[Image], search_dto: SearchRequestDTO) -> SearchResponseDTO:
        embeddings = self.embedder(imgs,[1 for _ in range(len(imgs))])
        embeddings = embeddings.unique(dim=1).squeeze(dim=1).cpu().tolist()

        neighbors = self.retrieval(search_dto, embeddings)

        llm_response = self.llm(neighbors)

        dto_parameters = [
            {"score": point.score, **point.payload}
            for point in neighbors
        ]

        return SearchResponseDTO(
            llm_response=llm_response,
            neighbors=[CarSearchResultDTO(**car) for car in dto_parameters]
        )