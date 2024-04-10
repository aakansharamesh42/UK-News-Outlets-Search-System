import React from 'react';
import { Container, Navbar, Nav, Row, Col, Card, Button } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import Markdown from 'markdown-to-jsx';

const privacyPolicyContent = `
# Privacy Policy

Last updated: 11th March 2024

## Introduction

At BritPress Navigator, we respect the privacy of our users. Our service is designed to enrich your online experience without compromising your personal data.

## Information We Collect

BritPress Navigator is structured to not collect any personal identifying information (PII). We collect data related to website links and necessary information to facilitate the functionality of our services.

- **Website Data:** We collect information about websites, including links and metadata, to provide our services.
- **Cookies:** If we use cookies, they are to maintain session integrity and enhance user experience. We do not use cookies to track personal information.

## How We Use Your Information

- **Service Provision:** The data we collect is used solely to perform the services requested by our users.
- **Improvements:** To enhance the platform's functionality and user interface.

## Data Sharing and Disclosure

- **No Sharing:** We do not share any user data, as we do not collect personal information.
- **Legal Compliance:** If at any point we are required by law to disclose any information, we will ensure that it is within the legal framework.

## Security

We prioritize the security of our users' interactions. While we do not collect personal data, any information gathered for the operation of our service is protected with industry-standard measures.

## Changes to Privacy Policy

Our Privacy Policy may change from time to time. We will provide notice on our website of any changes and update the 'last updated' date at the top of this policy.

## Contact Us

Should you have any queries regarding our privacy practices, please reach out to us.
`;


function PrivacyPolicy() {
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
       <Container>
            <Markdown>{privacyPolicyContent}</Markdown>
        </Container>
        </>
    );
}

export default PrivacyPolicy;
