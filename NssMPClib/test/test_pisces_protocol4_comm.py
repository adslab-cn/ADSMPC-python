import os
import queue
import sys
import threading

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from NssMPC.application.rag.pisces.protocol4 import (
    Protocol4Client,
    Protocol4Server,
    run_protocol4_client,
    run_protocol4_server,
)
from NssMPC.crypto.primitives.okvs import BinaryOKVS
from NssMPC.crypto.primitives.oprf import DHOPRFClient, DHOPRFParams, DHOPRFServer


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def describe_message(value):
    if hasattr(value, "table") and hasattr(value, "num_docs"):
        return (
            f"Protocol4PublicSetup(num_docs={value.num_docs}, slots={value.table.size}, "
            f"value_size={value.table.value_size}, method={value.table.method}, "
            f"prefix_size={value.prefix_size}, tf_size={value.tf_size})"
        )
    if hasattr(value, "elements"):
        first = hex(value.elements[0])[:34] if value.elements else "empty"
        return f"{type(value).__name__}(count={len(value.elements)}, first_prefix={first})"
    return type(value).__name__


class FakeParty:
    def __init__(self, incoming, outgoing, timeout=None):
        self.incoming = incoming
        self.outgoing = outgoing
        self.timeout = timeout

    def send(self, value):
        print(f"[FakeParty.send] sending {describe_message(value)}")
        self.outgoing.put(value)

    def receive(self):
        value = self.incoming.get(timeout=self.timeout)
        print(f"[FakeParty.receive] received {describe_message(value)}")
        return value


def test_protocol4_party_message_flow():
    section("Protocol 4 party message flow")
    queue_timeout = float(os.environ["PISCES_TEST_QUEUE_TIMEOUT"]) if "PISCES_TEST_QUEUE_TIMEOUT" in os.environ else None
    server_to_client = queue.Queue()
    client_to_server = queue.Queue()
    server_party = FakeParty(client_to_server, server_to_client, timeout=queue_timeout)
    client_party = FakeParty(server_to_client, client_to_server, timeout=queue_timeout)

    tf = torch.zeros(24, 7)
    tf[2, 0] = 3
    tf[2, 5] = 1
    tf[4, 3] = 2
    tf[9, 1] = 7
    tf[17, 6] = 5
    query = torch.tensor([2, 9, 13, 17])
    print("[Input] Server TF matrix shape=[vocab_size=24, num_docs=7]")
    print(tf)
    print(f"[Input] Client query tokens={query.tolist()}")
    print("[Expected message order] server sends setup -> client sends OPRF request -> server sends OPRF response -> client recovers TF")

    params = DHOPRFParams()
    okvs = BinaryOKVS(expansion=2.4, seed=b"protocol4-comm-okvs")
    server = Protocol4Server(okvs=okvs, oprf_server=DHOPRFServer(params=params, secret_key=13579))
    client = Protocol4Client(okvs=okvs, oprf_client=DHOPRFClient(params=params))

    server_error = []
    client_error = []
    client_result = []

    def server_thread():
        try:
            print("[Server thread] build setup, send setup, receive blinded request, send OPRF response")
            run_protocol4_server(server_party, server, tf)
        except Exception as exc:
            server_error.append(exc)

    def client_thread():
        try:
            print("[Client thread] receive setup, send blinded request, receive response, recover TF")
            result = run_protocol4_client(client_party, client, query)
            print("[Client thread] recovered TF")
            print(result)
            client_result.append(result)
        except Exception as exc:
            client_error.append(exc)

    t_server = threading.Thread(target=server_thread)
    t_client = threading.Thread(target=client_thread)
    t_server.start()
    t_client.start()
    join_timeout = queue_timeout if queue_timeout is not None else 120
    t_server.join(timeout=join_timeout)
    t_client.join(timeout=join_timeout)

    assert not t_server.is_alive()
    assert not t_client.is_alive()
    if server_error:
        raise server_error[0]
    if client_error:
        raise client_error[0]

    expected = tf[query].transpose(0, 1).float()
    print("[Expected TF]")
    print(expected)
    print("[Actual TF]")
    print(client_result[0])
    assert torch.equal(client_result[0], expected)
    print("[Check] fake-party message flow recovers exactly the expected TF")


def main():
    test_protocol4_party_message_flow()
    print("pisces protocol4 comm tests ok")


if __name__ == "__main__":
    main()
