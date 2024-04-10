import React, { useState, useEffect } from "react";
import {
  Container,
  Navbar,
  Nav,
  Button,
  Card,
  Spinner,
  Badge,
  Form,
  Row,
  Col,
} from "react-bootstrap";
import { useNavigate, Link } from "react-router-dom";
import "bootstrap/dist/css/bootstrap.min.css";
import QueryExpansion from "./queryExpansion";
import { fetchSearchResults, fetchQueryExpansions } from "./api";
import SentimentBadge from "./SentimentBadge";
import SearchBar from "./SearchBar";

function ResultsPage() {
  let navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filterYear, setFilterYear] = useState("all");
  const [sentimentFilter, setSentimentFilter] = useState("all");
  const [sourceFilter, setSourceFilter] = useState("all");
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPage, setTotalPage] = useState(1);
  const [useTime, setUseTime] = useState(0);
  const [scores, setScores] = useState([]);
  const [type, setType] = useState("tfidf" || "boolean");
  const [expansionTerms, setExpansionTerms] = useState([]);
  const maxPagesToShow = 5;

  function searchRoutine(searchParams) {
    const params = Object.fromEntries(searchParams.entries());
    const query = params.q;
    const type = params.type || "tfidf";
    setType(type);
    const page = parseInt(params.page) || 1;
    const expansions = parseInt(params.expansions) || 0;
    if (!query) {
      navigate("/");
    } else {
      let startTime = Date.now();
      setSearchQuery(query);
      if (expansions > 0 && type !== "boolean") {
        fetchQueryExpansions(query, expansions)
          .then((res) => {
            console.log(res);
            setExpansionTerms(res.added_terms);

            fetchSearchResults(res.expanded_query, type, page)
              .then((res) => {
                let elapsedTime = Date.now() - startTime;
                setUseTime(elapsedTime);
                setSearchResults(res.results || []);
                setTotalPage(res.total_pages);
                setCurrentPage(page);
                setScores(res.scores || []);
                setLoading(false);
              })
              .catch((error) => {
                alert("Error fetching search results:", error);
                navigate("/error");
              });
          })
          .catch((error) => {
            alert("Error fetching query expansions:", error);
            navigate("/error");
          });
      } else {
        fetchSearchResults(query, type, page)
          .then((res) => {
            let elapsedTime = Date.now() - startTime;
            setUseTime(elapsedTime);
            setSearchResults(res.results || []);
            setTotalPage(res.total_pages);
            setCurrentPage(page);
            setScores(res.scores || []);
            setLoading(false);
          })
          .catch((error) => {
            alert("Error fetching search results:", error);
            navigate("/error");
          });
      }
    }
  }

  useEffect(() => {
    const searchParams = new URLSearchParams(window.location.search);
    searchRoutine(searchParams);
  }, []);

  const handlePageChange = (page) => {
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("page", page);
    const newUrl = `${window.location.pathname}?${searchParams.toString()}`;
    window.location.href = newUrl;
  };

  const handleQueryExpansionSelect = (newQuery) => {
    setSearchQuery(newQuery);
    const searchParams = new URLSearchParams(window.location.search);
    searchParams.set("q", encodeURIComponent(newQuery));
    const newUrl = `${window.location.pathname}?${searchParams.toString()}`;
    window.location.href = newUrl;
  };

  const currentYear = new Date().getFullYear();
  const years = Array.from({ length: currentYear - 2020 + 1 }, (_, i) =>
    String(currentYear - i)
  );

  const uniqueSources = Array.from(
    new Set(searchResults.map((result) => result.source))
  );

  const filteredResults = searchResults.filter((result) => {
    const resultYear = new Date(result.date).getFullYear().toString();
    const meetsYearCriteria = filterYear === "all" || resultYear === filterYear;
    const meetsSourceCriteria =
      sourceFilter === "all" || result.source === sourceFilter;
    if (!meetsYearCriteria || !meetsSourceCriteria) return false;
    const sentiments = JSON.parse(result.sentiment);
    const { negative, neutral, positive } = sentiments.reduce(
      (acc, item) => {
        const parts = item.split(":");
        const type = parts[0].trim();
        const value = parseFloat(parts[1]);
        if (type === "positive") {
          acc.positive += value;
        } else if (type === "neutral") {
          acc.neutral += value;
        } else if (type === "negative") {
          acc.negative += value;
        }
        return acc;
      },

      { negative: 0, neutral: 0, positive: 0 }
    );

    const meetsSentimentCriteria =
      sentimentFilter === "all" ||
      (sentimentFilter === "positive" &&
        positive > neutral &&
        positive > negative) ||
      (sentimentFilter === "neutral" &&
        neutral > positive &&
        neutral > negative) ||
      (sentimentFilter === "negative" &&
        negative > positive &&
        negative > neutral);
    if (!meetsSentimentCriteria) return false;

    return true;
  });

  const ColorCodingGuide = () => (
    <div style={{ display: "flex", gap: "10px", marginBottom: "10px" }}>
      <Badge bg="success" text="dark">
        Positive
      </Badge>
      <Badge bg="secondary" text="dark">
        Neutral
      </Badge>
      <Badge bg="danger" text="dark">
        Negative
      </Badge>
    </div>
  );

  return (
    <>
      <Navbar bg="light" expand="lg">
        <Container>
          <Navbar.Brand as={Link} to="/">
          BritPress Navigator
          </Navbar.Brand>
          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="me-auto">
              <Nav.Link as={Link} to="/">
                Home
              </Nav.Link>
              <Nav.Link as={Link} to="/how-it-works">
                How It Works
              </Nav.Link>
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>
      <Container className="d-flex flex-column justify-content-center align-items-center py-4">
        <div className="text-center" style={{ width: "100%" }}>
          <SearchBar />
        </div>
      </Container>

      {!loading && expansionTerms.length > 0 && (
        <div className="d-flex justify-content-center align-items-center mb-3">
          <h6 style={{fontWeight: 800}}>Expanded query: {expansionTerms.join(", ")}</h6>
        </div>
      )}

      {/* disable when type is boolean */}
      {type !== "boolean" && (
        <QueryExpansion
          onQuerySelect={handleQueryExpansionSelect}
          currentQuery={searchQuery}
        />
      )}

      <Container>
        <Row>
          <Col md={3} className="filter-sidebar">
            <h4>Filter Results</h4>
            <Form>
              <Form.Group controlId="filterYear">
                <Form.Label>Year</Form.Label>
                <Form.Select
                  value={filterYear}
                  onChange={(e) => setFilterYear(e.target.value)}
                >
                  <option value="all">All Years</option>
                  {years.map((year) => (
                    <option key={year} value={year.toString()}>
                      {year}
                    </option>
                  ))}
                </Form.Select>
              </Form.Group>
              <Form.Group controlId="sentimentFilter">
                <Form.Label>Sentiment</Form.Label>
                <Form.Select
                  value={sentimentFilter}
                  onChange={(e) => setSentimentFilter(e.target.value)}
                >
                  <option value="all">All Sentiments</option>
                  <option value="positive">Positive</option>
                  <option value="neutral">Neutral</option>
                  <option value="negative">Negative</option>
                </Form.Select>
              </Form.Group>
              <Form.Group controlId="sourceFilter">
                <Form.Label>Source</Form.Label>
                <Form.Select
                  value={sourceFilter}
                  onChange={(e) => setSourceFilter(e.target.value)}
                >
                  <option value="all">All Sources</option>
                  {uniqueSources.map((source, index) => (
                    <option key={index} value={source}>
                      {source}
                    </option>
                  ))}
                </Form.Select>
              </Form.Group>
            </Form>
          </Col>
          <Col md={9}>
            <h2> Search Results </h2>
            {!loading && (
              <div>
                <h6>{`Search took ${
                  useTime / 1000
                } seconds, total ${totalPage} pages`}</h6>
              </div>
            )}
            <ColorCodingGuide />
            {loading ? (
              <Spinner animation="border" role="status">
                <span className="visually-hidden">Loading...</span>
              </Spinner>
            ) : filteredResults.length > 0 ? (
              filteredResults.map((result, index) => (
                <Card key={index} className="mb-3">
                  <Card.Body>
                    <Card.Title>{result.title}</Card.Title>
                    <SentimentBadge
                      sentiments={JSON.parse(result.sentiment).map((item) => {
                        const parts = item.split(":");
                        return {
                          type: parts[0].trim(),
                          value: parseFloat(parts[1]),
                        };
                      })}
                    />
                    <Card.Text>
                      {scores.length > 0 && (
                        <div>
                          <strong>Score:</strong>{" "}
                          {parseFloat(scores[index]).toFixed(3)}
                        </div>
                      )}
                      <strong>Topic: </strong> {result.topic}
                      <br />
                      <strong>Date:</strong> {result.date}
                      <br />
                      <strong>Summary:</strong> {result.summary}
                    </Card.Text>
                    <Button variant="primary" href={result.url}>
                      Visit source article
                    </Button>
                    <div>
                      <small className="text-muted">
                        Source: {result.source}, url:{" "}
                        <a href={result.url}>{result.url}</a>
                      </small>
                    </div>
                  </Card.Body>
                </Card>
              ))
            ) : (
              <div>No results found for the selected filters.</div>
            )}

            <div className="pagination-controls d-flex justify-content-center align-items-center mt-3">
              <Button
                variant="light"
                disabled={currentPage === 1}
                onClick={() => handlePageChange(currentPage - 1)}
                className="me-2"
              >
                Previous
              </Button>
              {Array.from(
                { length: Math.min(maxPagesToShow, totalPage) },
                (_, index) => {
                  const pageNumber = currentPage - 2 + index;
                  if (pageNumber < 1 || pageNumber > totalPage) {
                    return null;
                  }
                  return (
                    <Button
                      key={index}
                      variant={currentPage === pageNumber ? "primary" : "light"}
                      onClick={() => handlePageChange(pageNumber)}
                      className="me-2"
                    >
                      {pageNumber}
                    </Button>
                  );
                }
              )}
              <Button
                variant="light"
                disabled={currentPage === totalPage}
                onClick={() => handlePageChange(currentPage + 1)}
                className="me-2"
              >
                Next
              </Button>
            </div>
          </Col>
        </Row>
      </Container>
    </>
  );
}
export default ResultsPage;
