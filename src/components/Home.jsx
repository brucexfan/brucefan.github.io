import React, { useState } from 'react';
import { FaEnvelope, FaLinkedin } from 'react-icons/fa';
import { Link } from 'react-router-dom';

function Home() {
  const [showEmail, setShowEmail] = useState(false);

  const toggleEmail = () => {
    setShowEmail(!showEmail);
  };

  return (
    <div className="home">
      <h1>Adam Zweiger</h1>
      <div className="bio-container">
        <div className="profile-pic-container">
          <img src="/me.jpg" alt="Adam Zweiger" className="profile-pic" />
        </div>
        <div className="bio">
          <p>
            Hey! I'm sophomore at MIT studying CS (Course 6-3), interested in NLP and reasoning. 
            Currently, I'm working on two research projects related to test-time scaling.
            Previously, I've competed in olympiads in math, physics, and computing, studied a bit of pure math, and interned at AWS. 
            I'm also part of the HackMIT team, hoping to support tech innovation.
            Outside of academics, I love climbing, badminton, and tennis.
          </p> 
        </div>
      </div>
      <div className="contact-info">
        <div className="contact-link" onClick={toggleEmail}>
          <FaEnvelope className="contact-icon" />
          <span className="contact-text">
            {showEmail ? (
              <>
                ada<span style={{display: 'none'}}>foo</span>mz
                <span style={{display: 'none'}}>bar</span>@
                <span style={{display: 'none'}}>null</span>
                m<span style={{display: 'none'}}>foo</span>it.edu
              </>
            ) : (
              'Email'
            )}
          </span>
        </div>
        {/* <a href="https://www.linkedin.com/in/adam-zweiger-1b2b80200/" className="contact-link">
          <FaLinkedin className="contact-icon" />
          <span className="contact-text">LinkedIn</span>
        </a> */}
      </div>
    </div>
  );
}

export default Home;
