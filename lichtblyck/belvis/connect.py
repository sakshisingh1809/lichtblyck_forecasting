"""Module to connect to a Belvis tenant and query it for data."""

from abc import ABC, abstractmethod
from typing import Dict, Union, List
from requests.exceptions import ConnectionError
from pathlib import Path
import datetime as dt
import urllib
import jwt
import json
import requests


class BelvisConnection(ABC):

    _SERVER = "http://lbbelvis01:8040"

    @classmethod
    def from_usrpwd(cls, tenant, usr, pwd):
        return Connection_From_UsrPwd(tenant, usr, pwd)

    @classmethod
    def from_token(cls, tenant, usr):
        return Connection_From_Token(tenant, usr)

    def __init__(self, tenant: str):
        self._tenant = tenant
        self._lastsuccessfulquery = None

    tenant = property(lambda self: self._tenant)

    # To be implemented by children.

    @abstractmethod
    def authenticate(self):
        """Authenticate with server; raise PermissionError if details are incorrect."""
        ...

    @abstractmethod
    def request(self, url: str) -> requests.request:
        """Request data from the rest API at ``url``."""
        ...

    # Directly implemented at parent.

    def url(self, path: str, *queryparts: str) -> str:
        u = f"{self._SERVER}{path}"
        query = "&".join([urllib.parse.quote(qp, safe=":=") for qp in queryparts])
        return u if not query else f"{u}?{query}"

    def query_general(self, path: str, *queryparts: str) -> Union[Dict, List]:
        """Query connection for general information."""
        try:
            response = self.request(self.url(path, *queryparts))
        except ConnectionError as e:
            self._lastsuccessfulquery = None
            raise ConnectionError("Check VPN connection to Lichtblick.") from e

        if response.status_code == 200:
            self._lastsuccessfulquery = dt.datetime.now()
            return json.loads(response.text)
        elif self._lastsuccessfulquery is not None:
            # Query has worked before, so authentication might have expired. Try once.
            self._lastsuccessfulquery = None
            self.authenticate()
            return self.query_general(path, *queryparts)  # retry.
        else:
            raise RuntimeError(response)

    def query_timeseries(
        self, remainingpath: str, *queryparts: str
    ) -> Union[Dict, List]:
        """Query connection for timeseries information."""
        path = f"/rest/energy/belvis/{self.tenant}/timeSeries{remainingpath}"
        return self.query_general(path, *queryparts)


class Connection_From_UsrPwd(BelvisConnection):
    def __init__(self, tenant: str, usr: str, pwd: str):
        super().__init__(tenant)
        self.__usr = usr
        self.__pwd = pwd
        self._session = requests.Session()
        self.authenticate()

    def authenticate(self) -> None:
        # Check if usr/pwd is accepted by server.
        parts = (f"usr={self.__usr}", f"pwd={self.__pwd}", f"tenant={self.tenant}")
        url = self.url("/rest/session", *parts)
        response = self.request(url)
        if response.status_code != 200:
            raise PermissionError(f"Please check authentication details: {response}")

    def request(self, url: str) -> requests.request:
        """Create a request."""
        return self._session.get(url)


class Connection_From_Token(BelvisConnection):

    _AUTHFOLDER = Path(__file__).parent / "auth"

    def __init__(self, tenant: str, usr: str):
        super().__init__(tenant)
        self.__usr = usr
        self.authenticate()

    def authenticate(self) -> None:
        """Authentication with public-private key pair for user ``usr``."""

        # Open private key to sign token with.
        with open(self._AUTHFOLDER / "privatekey.txt", "r") as f:
            private_key = f.read()
        token = self.encode_with_token(self.__usr, private_key)

        # Open public key to decode the token with.
        with open(self._AUTHFOLDER / "publickey.txt", "r") as f:
            public_key = f.read()
        decoded_user = self.decode_with_token(token, public_key)
        if decoded_user != self.__usr:
            raise PermissionError(
                f"Username ({self.__usr}) and public key don't match."
            )

        # TODO: How to handle situation with multiple users?
        # TODO: Should token be located in the lichtblyck package? Or supplied by user via path?

        # Save details to be able to make queries.
        self._token = token

        # TODO: Check if authentication is accepted by server

    def encode_with_token(self, usr: str, key: str):
        # Create token that is valid for a given amount of time.
        claims = {
            "name": usr,
            "sub": self._tenant,
            "exp": dt.datetime.utcnow() + dt.timedelta(days=0, seconds=30),
            "iat": dt.datetime.utcnow(),
        }

        # "RSA 512 bit" in the PKCS standard for your client.
        return jwt.encode(payload=claims, key=key, algorithm="RS512")

    def decode_with_token(self, token, key):
        try:
            payload = jwt.decode(  # TODO: module jwt has no attribute 'decode'
                token, key=key, algorithms=["RS512"], options={"verify_signature": True}
            )
            return payload["name"]
        except jwt.ExpiredSignatureError:
            return "Signature expired. Please log in again."
        except jwt.InvalidTokenError:
            return "Invalid token. Please log in again."

    def request(self, url: str) -> requests.request:
        """Create a request."""
        return requests.get(url, token=self._token)
