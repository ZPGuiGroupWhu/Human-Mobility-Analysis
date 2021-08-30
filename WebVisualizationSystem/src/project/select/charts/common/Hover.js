import React, { Component } from 'react'

class Hover extends Component {
  constructor(props) {
    super(props);
    this.state = {
      isHovering: false,
      isClicked: false,
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
      console.log(this.state);
      this.setState({
        isClicked: false
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

export default Hover;