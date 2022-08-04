import React, { Component } from 'react';

class Hover extends Component {
  constructor(props) {
    super(props);
    this.defaultHoveringState = props.isHovering;
    this.defaultClickedState = props.isClicked;
    this.state = {
      isHovering: this.defaultHoveringState,
      isClicked: this.defaultClickedState,
    }
  }

  handleHovering = () => {
    this.setState({
      isHovering: true,
    })
  }

  cancelHovering = () => {
    this.setState({
      isHovering: false,
    })
  }

  onClick = () => {
    this.setState(prev => ({
      isClicked: !prev.isClicked
    }))
  }

  componentDidUpdate(prevProps, prevState) {
    if (prevProps.isReload !== this.props.isReload) {
      this.setState({
        isClicked: this.props.isClicked
      })
    }
  }

  render() {
    return (
      <div
        onMouseEnter={this.handleHovering}
        onMouseLeave={this.cancelHovering}
        onClick={this.onClick}
      >
        {this.props.children({ isHovering: this.state.isHovering, isClicked: this.state.isClicked })}
      </div>
    );
  }
}

Hover.defaultProps = {
  isHovering: false,
  isClicked: false,
}

export default Hover;