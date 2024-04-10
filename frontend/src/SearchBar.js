import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  InputGroup,
  FormControl,
  Button,
} from "react-bootstrap";
import { BsSearch } from "react-icons/bs";
import BASE_URL from "./api";
import debounce from "lodash.debounce";

function SearchBar() {
  const [searchQuery, setSearchQuery] = useState("");
  const [searchType, setSearchType] = useState("tfidf" || "boolean");
  const [errorMessage, setErrorMessage] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [validQuery, setValidQuery] = useState(true);
  const [numOfExpansions, setNumOfExpansions] = useState(1);

  const handleSearchClick = async () => {
    setErrorMessage("");
    if (!searchQuery.trim()) {
      setErrorMessage("Please enter a search query.");
      return;
    }

    // navigate the search page with url: /search?q=searchQuery.trim()&type=searchType&limit=10&page=1
    let params = new URLSearchParams();
    params.append("q", encodeURIComponent(searchQuery.trim()));
    params.append("type", searchType);
    params.append("limit", 10);
    params.append("page", 1);
    
    if (searchType !== "boolean") {
      params.append("expansions", numOfExpansions);
    }

    window.location.href = `/search-results?${params.toString()}`;
  };
  const fetchSuggestions = async (query) => {
    if (!query.trim()) {
      setSuggestions([]);
      return;
    }

    try {
      const response = await fetch(`${BASE_URL}/search/expand-query/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: query, num_expansions: 5 }), // Adjust num_expansions as needed
      });
      const data = await response.json();
      setSuggestions(data.expanded_queries);
    } catch (error) {
      console.error("Error fetching query suggestions:", error);
      setSuggestions([]);
    }
  };
  
  const debouncedFetchSuggestions = debounce(fetchSuggestions, 300);
  const handleChange = (e) => {
    const query = e.target.value;
    setSearchQuery(query);
    debouncedFetchSuggestions(query);
  };

  // --> clean up the debounced function on unmount
  useEffect(() => {
    return () => {
      debouncedFetchSuggestions.cancel();
    };
  }, []);

  const checkQuery = async (query) => {
    if (!query.trim()) {
      return;
    }
    
    if (searchType === "boolean") {
      const response = await fetch(`${BASE_URL}/search/validate-boolean-query?q=${encodeURIComponent(query)}`);
      // cast true or false to boolean
      const data = await response.json();
      setValidQuery(data);
      return;
    } else if (searchType === "tfidf") {
      setValidQuery(true);
    }
  }

  const debounceCheckQuery = debounce(checkQuery, 300);
  useEffect(() => {
    debounceCheckQuery(searchQuery);
    return () => {
      debounceCheckQuery.cancel();
    };
  }, [searchQuery, searchType]);

  return (
    <>
      {errorMessage && (
        <div style={{ color: "red", marginBottom: "10px" }}>{errorMessage}</div>
      )}{" "}
      {/* Display error message here */}
      <InputGroup className="mb-3">
        <FormControl
          placeholder="Search"
          aria-label="Search"
          value={searchQuery}
          onChange={handleChange} // Updated to use the new handleChange function                            spellCheck="true" // Enable spell check here
          autoComplete="on" // Enabled autocomplete here
          autoCorrect="on" // Enabled auto correct here
        />
        <select
          className="form-select"
          value={searchType}
          onChange={(e) => setSearchType(e.target.value)}
          style={{ maxWidth: "120px" }}
        >
          {/* <option value="standard">Standard</option> */}
          <option value="tfidf">TF-IDF</option>
          <option value="boolean">Boolean</option>
        </select>
        <Button variant="outline-secondary" onClick={handleSearchClick} disabled={!validQuery || !searchQuery.trim()}>
          <BsSearch />
        </Button>
        {/* input for integer for query expansion terms */}
        { searchType === "tfidf" && 
        <FormControl
          type="number"
          value={numOfExpansions}
          onChange={(e) => setNumOfExpansions(e.target.value)}
          style={{ maxWidth: "60px" }}
        /> }
      </InputGroup>
      <div style={{ color: "red", marginBottom: "10px" }}>
        {!validQuery && "Invalid query"}
      </div>
      {suggestions.length > 0 && (
        <ul className="list-group">
          {suggestions.map((suggestion, index) => (
            <li
              key={index}
              className="list-group-item list-group-item-action"
              onClick={() => {
                setSearchQuery(suggestion);
                setSuggestions([]); // Clear suggestions after selection
              }}
            >
              {suggestion}
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

export default SearchBar;
