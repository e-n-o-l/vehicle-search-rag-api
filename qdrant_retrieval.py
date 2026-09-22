from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Query, Range, FusionQuery, Fusion, Prefetch
from dtos import SearchRequestDTO


class QdrantRetrieval:

    def __init__(self, api_key: str, url: str, collection_name: str):

        self.qdrant_client = QdrantClient(
            api_key=api_key,
            url=url
        )

        self.collection_name = collection_name


    def __call__(self, search_dto: SearchRequestDTO, embeddings: list) -> list:
        parameters = search_dto.model_dump()
        requirements = [
            _build_field_condition(**key_val)
            for key_val in parameters.items()
        ]

        groups = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=Query(fusion=FusionQuery(
                    fusion=Fusion.RRF
            )),
            prefetch=[
                Prefetch(query=embedding, limit=5, filter=Filter(must=requirements))
                for embedding in embeddings
            ],
            limit=5
        )

        return groups.points



def _build_field_condition(key: str, value: any) -> FieldCondition:
    pair = {}

    if key == 'production_year':
        delta = 2
        pair[key] = Range(gte=value-delta, lte=value+delta)

    elif key == 'price':
        pair[key] = Range(gte=0, lte=value + value * 0.5)

    else:
        pair[key] = MatchValue(value=value)

    return FieldCondition(**pair)