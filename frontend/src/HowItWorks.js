import React from 'react';
import { Container, Navbar, Nav, Row, Col, Card, Button } from 'react-bootstrap';
import { Link } from 'react-router-dom';


function HowItWorks() {
    return (
        <>
            <Navbar bg="light" expand="lg">
                <Container>
                    <Navbar.Brand as={Link} to="/">BritPress Navigator</Navbar.Brand>
                    <Navbar.Toggle aria-controls="basic-navbar-nav" />
                    <Navbar.Collapse id="basic-navbar-nav">
                        <Nav className="me-auto">
                            <Nav.Link as={Link} to="/">Home</Nav.Link>
                            <Nav.Link as={Link} to="/how-it-works">How It Works</Nav.Link>
                        </Nav>
                    </Navbar.Collapse>
                </Container>
            </Navbar>

            <Container className="my-5">
                <Row className="justify-content-md-center">
                    <Col md={8}>
                        <h2 className="text-center mb-5">How It Works</h2>
                        <Card className="mb-3" border="light">
                            <Card.Body>
                                <Card.Title> Performing a Search</Card.Title>
                                <Card.Text>Type your query into the search bar. You have the choice between standard keyword searches for everyday inquiries or boolean searches for more complex, specific queries. For boolean searches, you can use operators like AND, OR, and NOT to refine your results</Card.Text>
                            </Card.Body>
                        </Card>

                        <Card className="mb-3" border="light">
			
                            <Card.Body>
                                <Card.Title> Advanced Search Algorithms</Card.Title>
                                <Card.Text>When you initiate a search, our service utilizes advanced algorithms to process your request. The backend employs Python scripts for web crawling, sentiment analysis, and our own word2vec models for contextual understanding.</Card.Text>
                            </Card.Body>
                        </Card>


                        <Card className="mb-3" border="light">
                            <Card.Body>
                                <Card.Title> Real-Time Processing</Card.Title>
                                <Card.Text>As you type, our system provides real-time suggestions to complete your query based on popular searches and past queries, thanks to our sophisticated query suggestion algorithm. This ensures you formulate effective searches quickly.</Card.Text>
                            </Card.Body>
                        </Card>


                        <Card className="mb-3" border="light">
                            <Card.Body>
                                <Card.Title> Results Generation and Categorization:</Card.Title>
                                <Card.Text>We use a blend of boolean logic and TF-IDF scoring to assess the relevance of documents. This process ranks your results not just by keyword matches but by the document's content value, determined by term frequency and the inverse frequency of the term in a corpus of documents.</Card.Text>
                            </Card.Body>
                        </Card>


                        
                        <Card className="mb-3" border="light">
                            <Card.Body>
                                <Card.Title> Results Presentation</Card.Title>
                                <Card.Text>The results are displayed on dedicated pages for different types of searches. We provide summarized information alongside each result to give you a quick overview of the content, saving you time in finding credible information.</Card.Text>
                            </Card.Body>
                        </Card>



                        
                        <Card className="mb-3" border="light">
                            <Card.Body>
                                <Card.Title> Corrections and Query Expansion</Card.Title>
                                <Card.Text>If a query contains a misspelling, our spell checker automatically suggests the correct form. Furthermore, we expand queries using synonyms and related terms to ensure comprehensive search coverage.</Card.Text>
                            </Card.Body>
                        </Card>


                        <Card className="mb-3" border="light">
                            <Card.Body>
                                <Card.Title> Sentiment Analysis</Card.Title>
                                <Card.Text>For each piece of content, our sentiment analysis tool gauges the sentiment of the text, which can be crucial in understanding the nature of the information provided, especially in news articles and social media posts</Card.Text>
                            </Card.Body>
                        </Card>



                        <Card className="mb-3" border="light">
                            <Card.Body>
                                <Card.Title> Unique Features</Card.Title>
                                <Card.Text>BritPress Navigator is more than just a search engine. It integrates a summarizer for quick understanding, spell checking, and autocomplete features, as well as sentiment analysis, all built on a solid infrastructure that leverages Redis for caching and robust index management for performance.</Card.Text>
                            </Card.Body>
                        </Card>
                    

                        <Card className="mb-3" border="light">
                            <Card.Body>
                                <Card.Title> Considerations for Use</Card.Title>
                                <Card.Text>Our service is cloud-native, ensuring scalability and reliability. It runs in Docker containers for seamless deployment. Our continuous integration and deployment processes, managed via GitHub Actions, allow us to maintain a high standard of service with minimal downtime.</Card.Text>
                            </Card.Body>
                        </Card>


                        <p>We believe in empowering our users with the tools to discern the truth. Experience the difference with BritPress Navigator now.</p>
                    </Col>
                </Row>
            </Container>
        </>
    );
}

export default HowItWorks;
