//
//  HomeTableViewCell.swift
//  lumos ios
//
//  Created by Shalin Shah on 7/18/17.
//  Copyright © 2017 Shalin Shah. All rights reserved.
//

import UIKit

class HomeTableViewCell: UITableViewCell {
    
    var background: UIView!
    let padding: CGFloat = 5

    @IBOutlet weak var examNumberLabel: UILabel!
    @IBOutlet weak var dateLabel: UILabel!
    
    override func awakeFromNib() {
        super.awakeFromNib()
        // Initialization code
        
        // Text style for examNumberLabel
        examNumberLabel.clipsToBounds = true
        examNumberLabel.alpha = 1
        examNumberLabel.text = "EXAM NUMBER".uppercased()
        examNumberLabel.font = UIFont(name: "AvenirNext-DemiBold", size: 12)
        examNumberLabel.textColor = UIColor(red: 1.00, green: 1.00, blue: 1.00, alpha: 1)
        
        
        // Text style for dateLabel
        dateLabel.clipsToBounds = true
        dateLabel.alpha = 1
        dateLabel.font = UIFont(name: "AvenirNext-Regular", size: 25)
        dateLabel.textColor = UIColor(red: 1.00, green: 1.00, blue: 1.00, alpha: 1)

    }
    
    override func setSelected(_ selected: Bool, animated: Bool) {
        super.setSelected(selected, animated: animated)

        // Configure the view for the selected state
    }

}
