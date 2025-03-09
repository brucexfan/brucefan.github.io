import React from 'react';

function Resume() {
  return (
    <div className="resume">
      <div className="resume-container">
      <iframe
          src="/Resume.pdf#navpanes=0"
          title="Resume"
          width="100%"
          height="100%"
          style={{border: 'none'}}
        />
      </div>
    </div>
  );
}

export default Resume;
