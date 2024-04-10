import React from 'react';
import { Container, Navbar, Nav, Row, Col, Card, Button } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import Markdown from 'markdown-to-jsx';

const TermsOfServiceContent = `
# Terms of Service

Last updated: 11th March 2024

## Agreement to Terms

By accessing or using BritPress Navigator, you agree to abide by these Terms of Service and all applicable laws and regulations governing the service. If you do not agree with any part of these terms, you are prohibited from using or accessing this site.

## Intellectual Property Rights

The content, features, functionality, and original materials provided on BritPress Navigator, including the user interface, are and shall remain the exclusive property of BritPress Navigator and its licensors. Our trademarks may not be used in connection with any product or service without the prior written consent of BritPress Navigator.

## Use License

- Permission is granted to temporarily download one copy of the materials on BritPress Navigator's website for personal, non-commercial transitory viewing only.
- This is the grant of a license, not a transfer of title, and under this license, you may not:
    - Modify or copy the materials;
    - Use the materials for any commercial purpose or for any public display;
    - Attempt to decompile or reverse engineer any software contained on BritPress Navigator's website;
    - Remove any copyright or other proprietary notations from the materials;
    - Transfer the materials to another person or "mirror" the materials on any other server.
- This license shall automatically terminate if you violate any of these restrictions and may be terminated by BritPress Navigator at any time.

## Disclaimer

The materials on BritPress Navigator's website are provided on an 'as is' basis. BritPress Navigator makes no warranties, expressed or implied, and hereby disclaims and negates all other warranties including, without limitation, implied warranties or conditions of merchantability, fitness for a particular purpose, or non-infringement of intellectual property or other violation of rights.

## Limitations

In no event shall BritPress Navigator or its suppliers be liable for any consequential damages arising out of the use or inability to use the materials on BritPress Navigator's website, even if BritPress Navigator or an authorized representative has been notified orally or in writing of the possibility of such damage.

## Modifications to Terms of Service

BritPress Navigator may revise these Terms of Service for its website at any time without notice. 
`;


function TermsOfService() {
    return (
        <>
            <Navbar bg="light" expand="lg">
                <Container>
                    <Navbar.Brand as={Link} to="/">BritPress Navigators</Navbar.Brand>
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
            <Markdown>{TermsOfServiceContent}</Markdown>
        </Container>
        </>
    );
}

export default TermsOfService;
