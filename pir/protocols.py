"""
Protocols and messages for Piano-like PIR schemes (client-dependent preprocessing).

This module defines:
1. Message protocols: Query, Response, EntryUpdate
2. Protocol interfaces: PIRClient, PIRServer, PIRParams

Piano-like PIR schemes (Piano, RMS24, Plinko, etc.) share these interfaces,
allowing different implementations to be used interchangeably.

Client-dependent preprocessing model:
- Offline phase: Server streams database to client; client generates hints
- Online phase: Client prepares query using a hint, server answers, client
  extracts result, then replenishes the consumed hint from backup hints
- Update: When database entries change, client must update affected hints
"""

from typing import Protocol, Iterator


# =============================================================================
# Message Protocols
# =============================================================================


class Query(Protocol):
    """
    Query from client to server.

    Concrete implementations define the query structure.
    """

    ...


class Response(Protocol):
    """
    Response from server to client.

    Concrete implementations define the response structure.
    """

    ...


class EntryUpdate(Protocol):
    """
    A single database entry update.

    Concrete implementations define how updates are represented.
    """

    ...


# =============================================================================
# Protocol Interfaces
# =============================================================================


class PIRClient(Protocol):
    """
    Protocol for PIR clients in the client-dependent preprocessing model.

    A PIR client must support:
    1. Offline phase: generate_hints() from database stream
    2. Online phase: query() -> extract() -> replenish_hints()
    3. Updates: update_hints() when database changes
    """

    def generate_hints(self, db_stream: Iterator[list[bytes]]) -> None:
        """
        Offline phase: process database and generate hints.

        Args:
            db_stream: Iterator yielding entries for each block
        """
        ...

    def query(self, indices: list[int]) -> list[Query]:
        """
        Prepare queries for database indices.

        Args:
            indices: List of database indices to retrieve

        Returns:
            List of queries to send to server
        """
        ...

    def extract(self, responses: list[Response]) -> list[bytes]:
        """
        Extract results from server responses.

        Args:
            responses: List of responses from server

        Returns:
            List of database entries at the queried indices
        """
        ...

    def replenish_hints(self) -> None:
        """
        Replenish consumed hints after extraction.

        Must be called after extract() to complete the query batch.
        """
        ...

    def update_hints(self, updates: list[EntryUpdate]) -> None:
        """
        Update hints affected by database changes.

        Args:
            updates: List of EntryUpdate from server
        """
        ...

    def remaining_queries(self) -> int:
        """
        Return number of queries remaining before offline phase needed.
        """
        ...


class PIRServer(Protocol):
    """
    Protocol for PIR servers.

    A PIR server must support:
    1. Streaming database for client's offline phase
    2. Answering online queries
    3. Updating entries and returning update info for client hints
    """

    def stream_database(self) -> Iterator[list[bytes]]:
        """
        Stream the database block by block.

        Used during the client's offline phase.

        Yields:
            Entries for each block, in order
        """
        ...

    def answer(self, queries: list[Query]) -> list[Response]:
        """
        Answer queries.

        Args:
            queries: List of queries from client

        Returns:
            List of responses
        """
        ...

    def update_entries(self, updates: dict[int, bytes]) -> list[EntryUpdate]:
        """
        Update multiple database entries.

        Args:
            updates: Mapping from index to new value

        Returns:
            List of EntryUpdate for client hint updates
        """
        ...


class PIRParams(Protocol):
    """
    Protocol for PIR parameters.

    Parameters define the structure of the database.
    """

    @property
    def num_entries(self) -> int:
        """Total number of database entries."""
        ...

    @property
    def entry_size(self) -> int:
        """Size of each entry in bytes."""
        ...
