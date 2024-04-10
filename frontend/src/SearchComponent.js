import React from "react";
import {
  Container,
} from "react-bootstrap";
import logoImage from "./logo.ico";
import SearchBar from "./SearchBar";
function SearchComponent(top_children, bottom_children) {

  return (
    <>
      <Container
        className="d-flex flex-column justify-content-center align-items-center"
        style={{ minHeight: "80vh" }}
      >
        <div className="text-center">
          <img
            src={logoImage}
            alt="BritPress Navigator Logo"
            style={{ maxWidth: "280px", width: "100%", marginBottom: "20px" }}
          />
          <SearchBar />
        </div>
      </Container>
    </>
  );
}

export default SearchComponent;
