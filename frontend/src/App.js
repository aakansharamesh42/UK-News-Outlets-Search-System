import React from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { Container, InputGroup, FormControl, Button, Navbar, Nav } from 'react-bootstrap';
import { BsSearch } from 'react-icons/bs';
import ResultsPage from './ResultsPage';
import ErrorPage from './ErrorPage';
import HowItWorks from './HowItWorks';
import PrivacyPolicy from './PrivacyPolicy';
import TermsOfService from './TermsOfService';
import logoImage from './logo.png';
import SearchComponent from './SearchComponent';

function App() {

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

            <SearchComponent />

            <footer className="text-center bg-light py-3">
                <Container>
                    © {new Date().getFullYear()} BritPress Navigator - All Rights Reserved
                    <div>
                    <a href="/privacy-policy">Privacy Policy</a> | <a href="/terms-of-service">Terms of Service</a>
                    </div>
                    
                </Container>
            </footer>
        </>
    );
}

function AppWrapper() {
    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<App />} />
                <Route path="/search-results" element={< ResultsPage />} />
                <Route path="/how-it-works" element={<HowItWorks />} />
                <Route path="/error" element={<ErrorPage />} />
                <Route path="/privacy-policy" element={<PrivacyPolicy />} />
                <Route path="/terms-of-service" element={<TermsOfService />} />
            </Routes>

        </BrowserRouter>
    );
}

export default AppWrapper;