//
//  FramesTableViewCell.swift
//  lumos ios
//
//  Created by Johnathan Chen on 2/25/18.
//  Copyright © 2018 Shalin Shah. All rights reserved.
//

import UIKit

class FramesTableViewCell: UITableViewCell {
    

    var frameImage: UIImageView!
    
    override func awakeFromNib() {
        super.awakeFromNib()
        // Initialization code
    }
    
    override init(style: UITableViewCellStyle, reuseIdentifier: String?) {
        super.init(style: style, reuseIdentifier: reuseIdentifier)
        
        frameImage = UIImageView(frame: CGRect.zero)
        contentView.addSubview(frameImage)
    
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func prepareForReuse() {
        super.prepareForReuse()
        
    }
    
    override func layoutSubviews() {
        super.layoutSubviews()
        setupContraints()
    }
    

    override func setSelected(_ selected: Bool, animated: Bool) {
        super.setSelected(selected, animated: animated)

        // Configure the view for the selected state
    }
    
    func setupContraints() {
        frameImage.translatesAutoresizingMaskIntoConstraints = false
        
        frameImage.topAnchor.constraint(equalTo: contentView.topAnchor).isActive = true
//        frameImage.centerXAnchor.constraint(equalTo: contentView.centerXAnchor).isActive = true
        frameImage.leftAnchor.constraint(equalTo: contentView.leftAnchor).isActive = true
        frameImage.rightAnchor.constraint(equalTo: contentView.rightAnchor).isActive = true
        frameImage.heightAnchor.constraint(equalToConstant: 300).isActive = true
        frameImage.widthAnchor.constraint(equalToConstant: 300).isActive = true
        frameImage.contentMode = .scaleAspectFit
        
        
        
//        let horizontalConstraint = NSLayoutConstraint(item: frameImage, attribute: NSLayoutAttribute.centerX, relatedBy: NSLayoutRelation.equal, toItem: contentView, attribute: NSLayoutAttribute.centerX, multiplier: 1, constant: 0)
//
//        let verticalConstraint = NSLayoutConstraint(item: frameImage, attribute: NSLayoutAttribute.centerY, relatedBy: NSLayoutRelation.equal, toItem: contentView, attribute: NSLayoutAttribute.centerY, multiplier: 1, constant: 0)
//
//        let widthConstraint = NSLayoutConstraint(item: frameImage, attribute: NSLayoutAttribute.width, relatedBy: NSLayoutRelation.equal, toItem: nil, attribute: NSLayoutAttribute.notAnAttribute, multiplier: 1, constant: 300)
//
//        let heightConstraint = NSLayoutConstraint(item: frameImage, attribute: NSLayoutAttribute.height, relatedBy: NSLayoutRelation.equal, toItem: nil, attribute: NSLayoutAttribute.notAnAttribute, multiplier: 1, constant: 300)
        
    }

}
